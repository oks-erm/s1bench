"""SHVE package preflight, transparent input-only baselines and restricted run CLI."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import csv
from contextlib import contextmanager
import hashlib
import io
import os
import json
from pathlib import Path
import re
import threading
import time
import zipfile

import benchmark as b
import decision_analysis as d

MODELS = ('jev', 'gpt', 'nimble', 'laya')
TASKS = ('sector_imputation', 'semantic_validation', 'label_qa', 'entity_match', 'vision_routing', 'data_gap_identification')
BASELINE_VERSION = 'shve-rules-v1'
# Transparent vocabulary selected from development descriptions and category meanings.
# No row identifiers or evaluation labels are used by the predictor.
WORDS = {'Sector 1': ('industrial', 'factory', 'kiln'), 'Sector 2': ('domestic', 'residential', 'household', 'private home'),
         'Sector 3': ('hotel', 'restaurant', 'catering'), 'Sector 4': ('farm', 'crop', 'agricultur'),
         'Sector 5': ('public authority', 'municipal', 'government'), 'Sector 6': ('transport', 'fleet'),
         'Sector 7': ('aerosol',), 'Sector 9': ('reseller',)}


def unpack(archive, destination):
    destination = Path(destination)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (destination/member.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or member.file_size > 32*1024*1024:
                raise ValueError('Unsafe archive path or oversized member.')
            if member.is_dir():
                continue
            payload = z.read(member)
            if target.exists() and target.read_bytes() != payload:
                raise ValueError('Refusing to overwrite different handoff data: '+str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    return destination/'SHVE_Codex_Handoff'


def jsonl(path):
    return [b.parse(line) for line in Path(path).read_text('utf-8-sig').splitlines() if line.strip()]


def preflight(root):
    root = Path(root)
    sums = b.parse((root/'SHA256SUMS.json').read_text())
    for filename, expected in sums.items():
        path = root/filename
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Unsafe checksum path.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Checksum mismatch: '+filename)
    native = jsonl(root/'native/benchmark_data.jsonl')
    revised = jsonl(root/'revised/benchmark_data.jsonl')
    original = jsonl(root/'original/benchmark_data.jsonl')
    inputs = jsonl(root/'revised/inputs_only.jsonl')
    payloads = jsonl(root/'native/model_payloads.jsonl')
    if not (len(native) == len(revised) == len(original) == len(inputs) == len(payloads) == 1200):
        raise ValueError('Expected exactly 1,200 cases in each combined representation.')
    by_id = {c['id']: c for c in revised}
    reviews = {r['id']: r for r in csv.DictReader(io.StringIO((root/'revised/review_labels.csv').read_text()))}
    if len(by_id) != 1200 or len(reviews) != 1200:
        raise ValueError('Duplicate or missing IDs.')
    for task in TASKS:
        mirror = jsonl(root/f'revised/{task}.jsonl')
        if mirror != [r for r in revised if r['task'] == task]:
            raise ValueError('Per-task mirror mismatch: '+task)
    families = defaultdict(list)
    source_ids, target_leaks = [], []
    for n, r, o, i, payload in zip(native, revised, original, inputs, payloads):
        if not (n['id'] == r['id'] == o['id'] == i['id']):
            raise ValueError('Case order/ID mismatch.')
        if n['expected'] != r['gold'] or r['gold'] != o['gold'] or reviews[r['id']]['reviewed_gold'] != r['gold']:
            raise ValueError('Gold/review mismatch: '+r['id'])
        if n['input'] != r['state'] or n['cluster_id'] != r['family_id'] or n['split'] != r['split']:
            raise ValueError('Native mapping mismatch.')
        if list(n['question']['criteria']) != r['choices'] or n['critical_error_choices'] != r['critical_error_choices']:
            raise ValueError('Choice order or risk metadata lost.')
        if n['question']['instructions'] != r['question']+'\nRubric:\n'+json.dumps(r['rubric'],ensure_ascii=False):
            raise ValueError('Rubric mapping mismatch.')
        if i != {k:r[k] for k in ('id','task','type','state','question','rubric','choices')}:
            raise ValueError('Input-only mismatch.')
        if payload != {'state': n['input'], 'questions': {'answer': n['question']}}:
            raise ValueError('Payload mismatch.')
        state = b.parse(n['input']) if isinstance(n['input'], str) else n['input']
        b.canonical(state)  # Reject nonfinite values even inside JSON-formatted text.
        if re.search(r'"(?:ChurnNextMonth|gold|expected|gold_reason|review_status|family_id|scenario)"\s*:', json.dumps(state), re.I):
            target_leaks.append(n['id'])
        if 'record_id' in state.get('record', {}): source_ids.append(n['id'])
        families[n['cluster_id']].append(n)
    if target_leaks:
        raise ValueError('Exposed evaluation metadata in inputs: '+str(target_leaks[:5]))
    for family, members in families.items():
        if len(members) != 2 or len({m['expected'] for m in members}) != 1 or len({m['split'] for m in members}) != 1:
            raise ValueError('Paired-family/split inconsistency: '+family)
        if b.input_json(members[0]['input']) == b.input_json(members[1]['input']):
            raise ValueError('Representation pair has identical transmitted state: '+family)
    cfg = b.default_config()
    cases = b.load_jsonl((root/'native/benchmark_data.jsonl').read_bytes(), 'SHVE', cfg)
    for task in TASKS:
        counts = Counter(c['split'] for c in native if c['task'] == task)
        if counts != {'development':40, 'evaluation':160}: raise ValueError('Unexpected splits: '+task)
    for split in ('development','evaluation'):
        if jsonl(root/f'native/{split}.jsonl') != [r for r in native if r['split'] == split]:
            raise ValueError('Split mirror mismatch.')
    changed = [n['id'] for n, o in zip(revised, original) if n['state'] != o['state']]
    logged = {x['id'] for x in b.parse((root/'data_changes.json').read_text())}
    if set(changed) != logged or len(changed) != 210: raise ValueError('Repair log mismatch.')
    checks = {'revision':'shve-v2-20261006','cases':len(cases),'families':len(families),
              'splits':dict(Counter(c['split'] for c in cases)), 'checksum_files':len(sums),
              'prior_repaired_cases':len(changed),'additional_input_repairs':0,'gold_changes':0,
              'source_id_fields_present':len(source_ids),'target_metadata_leaks':0,
              'checks':'hashes, IDs, loader, risk/gold membership, original gold, review CSV, mirrors, payload mapping, split/family separation, nonfinite values, distinct representation pairs',
              'warnings':['Source-grounded scenario benchmark with synthetic policies; labels are AI-reviewed, not human-approved.',
                          'Synthetic record IDs remain in inputs; no baseline may use them. Source-ID shortcuts are a potential limitation.',
                          'Repeated decision templates limit semantic diversity. Source-entity independence is unverified.',
                          '160 evaluation cases / 80 declared families per task; production recommendation gates remain unmet.',
                          'Matching litres, sector dictionary, history precedence and routing rules are constructed premises, not SHVE-approved production policies.'],
              'diagnostics': d.diagnostics(cases, [])}
    return cases, checks


def sector(record):
    text = str(record.get('activity_note', '')).lower().split('; imported text')[0]
    if any(w in text for w in ('primary activity unknown', 'not recorded', 'mixed', 'unknown')):
        return 'CLARIFY'
    labels = [label for label, words in WORDS.items() if any(w in text for w in words)]
    return labels[0] if len(labels) == 1 else 'CLARIFY'


def month(value):
    match = re.match(r'^(\d{4})-(\d{2})', str(value))
    if not match or not 1 <= int(match[2]) <= 12:
        raise ValueError('Invalid month')
    return int(match[1])*12+int(match[2])-1


def baseline(task, state, question):
    """Accepts model-visible state/question only; never a full case or gold label."""
    state = b.parse(state) if isinstance(state, str) else state
    r = state.get('record', {})
    if task in {'sector_imputation', 'label_qa'}:
        label = sector(r)
        return label if task == 'sector_imputation' or label == 'CLARIFY' else ('KEEP' if r.get('Sector') == label else 'CORRECT')
    if task == 'semantic_validation':
        keys = ('MinFillLevel','MaxFillLevel','MaxCapacity')
        units = [r.get('unit_metadata', {}).get(k) for k in keys]
        try:
            values = [float(r[k]) for k in (*keys, 'negative_ratio')]
        except (KeyError, TypeError, ValueError):
            return 'CLARIFY'
        if any(isinstance(r[k], bool) for k in (*keys,'negative_ratio')) or not all(d.finite(x) for x in values) or not all(units) or len(set(units)) != 1:
            return 'CLARIFY'
        low, high, cap, ratio = values
        return 'VALID' if 0 <= low <= high <= cap and 0 <= ratio <= 1 else 'INVALID'
    if task == 'entity_match':
        left, right = state.get('left', {}), state.get('right', {})
        a, z = left.get('delivery_point_ref'), right.get('delivery_point_ref')
        if a is not None and z is not None and a != z: return 'NO_MATCH'
        if a is None or z is None: return 'CLARIFY'
        x, y = left.get('site_token'), right.get('site_token')
        return 'CLARIFY' if x is not None and y is not None and x != y else 'MATCH'
    if task == 'vision_routing':
        text = str(state.get('request','')).lower()
        if any(w in text for w in ('merge','deletion','disputed')): route='HUMAN_REVIEW'
        elif any(w in text for w in ('numeric','columns','data types','schema','format')): route='DETERMINISTIC'
        elif any(w in text for w in ('forecast','churn','trained checkpoint')): route='PREDICTIVE'
        elif any(w in text for w in ('explain','remediation','open-ended')): route='LLM'
        elif any(w in text for w in ('classify','sector','activity note','flag')): route='SYSTEM_ONE'
        else: route='HUMAN_REVIEW'
        return route if route == 'HUMAN_REVIEW' or route in state.get('approved_local_routes',[]) else 'HUMAN_REVIEW'
    if task == 'data_gap_identification':
        if any(r.get(k) in (None, '') for k in state.get('required_fields', ['Sector','MaxCapacity'])):
            return 'MISSING_REQUIRED_FIELD'
        try:
            now = month(state['as_of_month'])
            observed = {month(m) for m in state['observed_months']}
            if not observed: return 'CLARIFY'
            if set(range(min(observed),now+1))-observed: return 'MISSING_PERIOD'
            if now-month(state['latest_source_month']) > 1: return 'STALE_SOURCE'
            creation = state.get('account_created_month', r.get('account_created_month'))
            age = now-month(creation)+1 if creation else r.get('tenure_months')
            if not d.finite(age): return 'CLARIFY'
            if age < state.get('required_history_months',6): return 'INSUFFICIENT_HISTORY'
        except (KeyError, TypeError, ValueError):
            return 'CLARIFY'
        if any(state.get(k) != 'complete' for k in ('delivery_feed_status','support_feed_status')): return 'CLARIFY'
        return 'NO_GAP'
    raise ValueError('No SHVE baseline for '+task)


def baseline_records(cases, cfg):
    rows = []
    for c in cases:
        started = time.perf_counter()
        value = baseline(c['task'], b.state_for(c), c['question'])
        latency = time.perf_counter()-started
        valid, correct, numeric, decision = b.grade(value,c,cfg)
        rows.append({'model':'rules','id':c['id'],'dataset':c['dataset'],'task':c['task'], 'cluster_id':c['cluster_id'],
                     'type':c['question']['type'],'expected':c['expected'],'value':value,'decision':decision,
                     'valid':valid,'correct':correct,'numeric_error':numeric,'api_error':None,'critical':False,
                     'critical_error_choices':c.get('critical_error_choices',[]),'phase':'primary','scored':True,
                     'latency_s':latency,'latency_basis':'in-process rule execution, not HTTP end-to-end',
                     'resolved_model':BASELINE_VERSION,'confidence':None,'probabilities':None,
                     'input_tokens':0,'output_tokens':0,'_api_cost':0.0})
    return rows


def run_config(existing):
    cfg = copy.deepcopy(existing)
    indexed = {m['name']:m for m in cfg['models']}
    if any(name not in indexed for name in MODELS): raise ValueError('All four configured models are required.')
    cfg['models'] = [indexed[name] for name in MODELS]
    for m in cfg['models']: m['enabled'] = True
    cfg['protocol'] = {**b.default_config()['protocol'], **cfg.get('protocol',{}),
                       'warmup_calls':2,'repeat_cases':100,'repetitions':3,'max_requests':5608,'retry_allowance':0}
    cfg['business'] = {**b.default_config()['business'], **cfg.get('business',{}), 'baseline':'gpt'}
    cfg['tasks'] = {}  # Old generic task IDs overlap SHVE, but their typed policies differ.
    for task in TASKS:
        cfg['tasks'][task] = {'acceptance':{'mode':'label','review_labels':d.REVIEW_LABELS,
                             'business_approved':False,'ineligible_labels':['MATCH'] if task=='entity_match' else []}}
    cfg['shve_protocol'] = {'dataset_revision':'shve-v2-20261006','risk_policy':d.RISK_VERSION,
                            'source':'source-grounded scenarios with synthetic policies','retry_allowance':0,
                            'threshold_rule':'Development only: lowest probability threshold with >=10 accepted families and zero observed accepted errors; otherwise all review. Provisional, not certified calibration.',
                            'baseline_version':BASELINE_VERSION,'baseline_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return cfg


def freeze_thresholds(records, cases, cfg):
    index = {c['id']:c for c in cases}
    policies = {}
    for alias in MODELS:
        policies[alias] = {}
        for task in TASKS:
            dev = [r for r in records if b.primary_row(r) and r['model']==alias and r['task']==task and index[r['id']]['split']=='development']
            base = {**d.policy_for(cfg,task), 'mode':'probability'}
            chosen = None
            for t in (0,.5,.6,.7,.8,.9,.95,.99,1):
                p = {**base,'threshold':t}
                accepted = [r for r in dev if d.acceptance(r,p)=='accepted']
                if len({r['cluster_id'] for r in accepted}) >= 10 and all(r['correct'] for r in accepted):
                    chosen = p; break
            if chosen is None:
                chosen = {**base,'threshold':1,'accept_none':True}
            policies[alias][task] = {**chosen,'selection_basis':'development only; provisional; evaluation never used',
                                    'development_cases':len(dev),'development_ids_sha256':b.fingerprint(sorted(r['id'] for r in dev))}
    return policies


def capture_identity(model):
    """Read metadata from the configured local endpoint; never pull model weights."""
    from urllib.request import urlopen, Request
    from urllib.parse import urlsplit
    if model['name'] != 'nimble':
        return
    url = urlsplit(model['endpoint'])
    base = f'{url.scheme}://{url.netloc}'
    evidence = {'status':'unavailable','requested_alias':model['model'],'captured_utc':b.utc()}
    try:
        with urlopen(base+'/api/tags',timeout=5) as response: tags=json.load(response)
        match = next((m for m in tags.get('models',[]) if m.get('name') in {model['model'],model['model']+':latest'}),None)
        if match and match.get('digest'):
            evidence.update(status='captured',digest=match['digest'],details=match.get('details'),source='/api/tags')
        request = Request(base+'/api/show',data=json.dumps({'model':model['model']}).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=5) as response: info=json.load(response)
        evidence.update(details=info.get('details'),parameters=info.get('parameters'),capabilities=info.get('capabilities'),model_info=info.get('model_info'))
    except (OSError, ValueError) as exc:
        evidence['unavailable_reason']=type(exc).__name__
    model['identity_evidence']=evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--handoff',type=Path,default=b.BASE/'data/SHVE_Codex_Handoff')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--recover-from',type=Path,help='Recover incomplete models using the exact saved protocol')
    parser.add_argument('--only',nargs='+',choices=MODELS,help='Only with --recover-from: models to recover')
    parser.add_argument('--egress-approved',action='store_true')
    parser.add_argument('--report',type=Path,help='Regenerate analysis only, no calls')
    args=parser.parse_args()
    if args.report:
        export_analysis(args.report);return
    root=unpack(args.archive,b.BASE/'data') if args.archive else args.handoff
    cases,checks=preflight(root)
    cfg=run_config(b.read_config())
    if args.recover_from:
        previous=b.read_report(args.recover_from)
        errors,_=b.audit_report(previous)
        if errors:raise ValueError('Source evidence is invalid')
        cfg=copy.deepcopy(previous['config'])
        if b.input_json(sorted(cases,key=lambda c:c['id'])) != b.input_json(sorted(previous['cases'],key=lambda c:c['id'])):
            raise ValueError('Recovery data differs from source')
        if not args.only:raise ValueError('Choose incomplete models with --only')
        current={m['name']:m for m in b.read_config()['models']}
        for m in cfg['models']:
            source=current[m['name']]
            if (m.get('model'),m.get('endpoint'))!=(source.get('model'),source.get('endpoint')):
                raise ValueError('Configured provider/model differs from frozen recovery profile')
            for key in ('api_key','api_key_env','headers'):
                if key in source:m[key]=copy.deepcopy(source[key])
            if m['name'] in args.only:
                completed=[r for r in previous['records'] if r['model']==m['name'] and b.primary_row(r)]
                if len(completed)==len(cases):raise ValueError('Refusing to rerun a complete model')
    elif args.only:
        raise ValueError('--only requires --recover-from')
    b.validate_data(cases,cfg)
    selected = args.only or list(MODELS)
    for model in cfg['models']:
        model['enabled'] = True
    output=b.BASE/'results/shve-preflight'
    output.mkdir(parents=True,exist_ok=True)
    b.write_json(output/'data_checks.json',checks)
    b.write_json(output/'protocol.plan.json',cfg['shve_protocol'])
    rules=baseline_records(cases,cfg)
    (output/'baseline.raw.jsonl').write_bytes(b.jsonl_bytes(rules))
    plan=b.request_plan(cases,cfg,selected=selected)
    if plan['maximum_requests']!=1402*len(selected) or [m['name'] for m in cfg['models']]!=list(MODELS):raise ValueError('Unexpected request plan')
    public={'request_plan':plan,'recovery_source':str(args.recover_from) if args.recover_from else None,
            'models':[{k:m.get(k) for k in ('name','model','endpoint','pricing_per_million')} for m in cfg['models'] if m['name'] in selected],
            'cost_estimate':'Unknown until exact rates and token usage are available; zero automatic retries.',
            'credentials_available':{m['name']:bool(b.model_key(m)) for m in cfg['models'] if not b.local_model(m)}}
    b.write_json(output/'request_plan.json',public)
    print(json.dumps(public,indent=2),flush=True)
    if not args.execute:return
    if not args.egress_approved:raise ValueError('Actual dataset egress approval required separately from scenario policies.')
    missing=[m['name'] for m in cfg['models'] if not b.local_model(m) and not b.model_key(m)]
    if missing:print('Hosted models will be recorded as skipped until keys are configured: '+', '.join(missing),flush=True)
    cfg['shve_protocol']['egress_approval']='User approved configured Jev and OpenAI endpoints for this package, 2026-10-06'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import local_runtime
    cancel=threading.Event()
    @contextmanager
    def session(model):
        if model['name'] not in MODELS:raise ValueError('Excluded model cannot start')
        with local_runtime.model_session(model,cancel):
            capture_identity(model)
            yield
    def progress(event):
        if event.get('row') and event['issued']%100==0:print(f"{event['issued']}/{event['maximum']}",flush=True)
        elif not event.get('row') and event.get('message') and ' · ' not in event['message']:print(event['message'],flush=True)
    result=b.run(cfg,cases,model_session=session,cancel=cancel,progress=progress,selected=selected)
    folder=Path(result['folder'])
    for filename in ('data_checks.json','protocol.plan.json','request_plan.json'):
        (folder/filename).write_bytes((output/filename).read_bytes())
    (folder/'data_changes.json').write_bytes((root/'data_changes.json').read_bytes())
    export_analysis(folder)
    print(json.dumps(result),flush=True)


def export_analysis(folder):
    """Export derived artifacts while preserving frozen inputs and model responses."""
    import report_export
    folder=Path(folder)
    report=b.read_report(folder)
    if not report['config'].get('shve_protocol'):
        raise ValueError('SHVE analysis requires a saved SHVE protocol.')
    evidence_files=('raw.jsonl','data.snapshot.jsonl','config.snapshot.json','manifest.json')
    frozen={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in evidence_files}
    cfg=copy.deepcopy(report['config'])
    cfg['frozen_acceptance']=freeze_thresholds(report['records'],report['cases'],cfg)
    b.write_json(folder/'thresholds.analysis.json',cfg['frozen_acceptance'])
    payload=report_export.build_payload(report,analysis_cfg=cfg)
    (folder/'report.html').write_text(report_export.render_report(payload),encoding='utf-8')
    summaries=payload['summaries_by_reference'][payload['baseline']]
    decision=payload['decision_analysis']
    b.write_json(folder/'decision.analysis.json',summaries)
    b.write_json(folder/'decision_diagnostics.json',decision['diagnostics'])
    b.write_json(folder/'probability_quality.json',[{k:r[k] for k in ('model','dataset','task','probability_quality')} for r in summaries])
    def expanded(field):
        return [{'model':r['model'],'dataset':r['dataset'],'task':r['task'],**entry} for r in summaries for entry in r[field]]
    tables={'per_task_metrics.csv':summaries,'per_label.csv':expanded('per_label'),
            'error_directions.csv':expanded('error_directions'),'risk_curves.csv':decision['risk_curves'],
            'baseline_summary.csv':decision['baseline_summaries'],
            'business_decisions.csv':decision['baseline_comparisons']}
    for name,rows in tables.items():(folder/name).write_bytes(b.csv_bytes(rows))
    (folder/'baseline.raw.jsonl').write_bytes(b.jsonl_bytes(decision['baseline_records']))
    (folder/'business_summary.md').write_text(business_summary(report,payload),encoding='utf-8')
    if any(hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest for name,digest in frozen.items()):
        raise RuntimeError('Analysis unexpectedly changed frozen evidence.')
    b.write_json(folder/'analysis_provenance.json',{'frozen_evidence_sha256':frozen,
                 'basis':'Saved-response analysis; no inference calls or label changes',
                 'baseline_version':BASELINE_VERSION,'threshold_basis':'Development responses only, provisional',
                 'generated_utc':b.utc()})
    b.seal(folder)


def business_summary(report,payload):
    """A concise saved-evidence handoff; unknown and partial results stay explicit."""
    summaries=payload['summaries_by_reference'][payload['baseline']]
    records=report['records']
    lines=[f"# SHVE business evidence — {Path(report['folder']).name}",
           '',f"Dataset revision: {report['config']['shve_protocol'].get('dataset_revision','unknown')}.",
           'Source-grounded constructed scenarios with AI-reviewed labels and proposed policies. Development and evaluation are separate. Production recommendations remain inconclusive.',
           '', '## Execution']
    for alias in report['manifest'].get('selected_models',[]):
        rows=[r for r in records if r['model']==alias]
        primary=[r for r in rows if b.primary_row(r)]
        lines.append(f"- {alias}: {len(primary)}/{len(report['cases'])} primary cases; {sum(r.get('phase')=='repeat' for r in rows)} repeat requests; {sum(r.get('phase')=='warmup' for r in rows)} warm-ups; {sum(bool(r.get('api_error')) for r in rows)} API errors. Availability: {report.get('availability',{}).get(alias,{}).get('status','unknown')}.")
    lines.extend(['','## Evaluation findings','', '| Use case | Model | Status | Task success | Directional errors / exposed cases | Threshold accepted / planned |',
                  '|---|---|---|---|---|---|'])
    for row in summaries:
        if row['evaluation_scope']!='evaluation':continue
        accuracy=f"{row['success_rate']:.1%}" if row['status']=='complete' and d.finite(row['success_rate']) else 'unavailable — incomplete'
        lines.append(f"| {row['task']} | {row['model']} | {row['status']} | {accuracy} | {row['critical_failures']} / {row.get('risk_exposures',0)} | {row['acceptance']['accepted']} / {row['planned']} |")
    lines.extend(['','## Comparison and business decision',
                  'Use business_decisions.csv for model-minus-rule paired family intervals and GPT remains the paired model reference in the HTML viewer. Partial cohorts have no incremental-benefit estimate. Numeric/unit checks, authoritative matching and explicit data-gap rules should be assessed against deterministic implementation before paying for neural classification.',
                  'Sector/label stewardship needs demonstrated incremental benefit on human-adjudicated unseen descriptions. Entity MATCH is a human-review candidate. Vision scores measure policy classification; downstream execution, routing savings and churn uplift were not measured.',
                  '', '## Costs and workflow assumptions',
                  'Recorded call costs, including warm-ups/repeats and any priced failed calls:',
                  '```json',json.dumps(payload['cost_totals'],indent=2,ensure_ascii=False),'```',
                  'Unknown token usage or rates stay unknown. Local API charges of zero exclude hosting and TCO. Workflow volume, manual/review/audit/rework time, labour rates, setup, hosting and FX are editable assumptions in report.html. Missing assumptions prevent a complete savings estimate; constructed class prevalence is not operational workload.',
                  '', '## Next evidence',
                  'Human-adjudicated confirmation data with unseen templates, verified source independence, representative multilingual and missingness cases, rare-sector/risk exposure, declared risk/latency/repeat targets, organisational deployment approval and measured review/audit costs.',
                  'The evaluation has 160 cases / 80 declared families per use case. Repeats and paired variants do not increase independent sample size. Perfect observed samples and provisional development thresholds do not establish zero future risk or production calibration.'])
    return '\n'.join(lines)+'\n'


if __name__=='__main__':main()
