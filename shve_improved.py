"""Frozen improved SHVE suite: semantic judgments and weighted supplied-label churn.

This module reuses the benchmark evidence/locking/runtime layers. Preparation and
analysis never call models. The legacy handoff protocol remains in shve.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import benchmark as b
import churn_analysis as churn
import decision_analysis as d
import shve
import shve_cases

MODELS = shve.MODELS
TASKS = tuple(shve_cases.POLICIES) + ('churn_prediction',)
BASELINE_VERSION = 'shve-lexical-and-historical-prior-v1-20261007'
DEFAULT_DATA = b.BASE / 'data/shve-improved-20261007'


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_config(existing, cases):
    cfg = copy.deepcopy(existing)
    profiles = {m['name']: m for m in cfg['models']}
    if any(alias not in profiles for alias in MODELS):
        raise ValueError('The four existing Jev/GPT/Nimble/Laya profiles are required.')
    cfg['models'] = [profiles[alias] for alias in MODELS]
    for model in cfg['models']:
        model['enabled'] = True
        if model['name'] == 'gpt':
            model['structured_output'] = True
    cfg['protocol'] = {**b.default_config()['protocol'], **cfg.get('protocol', {}),
                       'warmup_calls': 2, 'repeat_cases': 100, 'repetitions': 3,
                       'batches': 10, 'max_requests': (len(cases)+202)*len(MODELS), 'retry_allowance': 0}
    cfg['business'] = {**b.default_config()['business'], **cfg.get('business', {}), 'baseline': 'gpt'}
    cfg['tasks'] = {task: {'acceptance': {'mode': 'label', 'review_labels': [*d.REVIEW_LABELS, 'UNKNOWN'],
                          'business_approved': False, 'ineligible_labels': ['MATCH'] if task=='entity_match' else []}}
                    for task in TASKS}
    cfg['tasks']['churn_prediction'] = {'label_semantics': 'observed_outcome', 'threshold': .5,
                                      'acceptance': {'mode': 'probability', 'threshold': .5,
                                                     'business_approved': False}}
    cfg['shve_protocol'] = {'suite': 'improved', 'dataset_revision': 'shve-improved-v1-20261007',
                           'risk_policy': d.RISK_VERSION, 'retry_allowance': 0,
                           'input_sha256': b.fingerprint(sorted(cases, key=lambda c:c['id'])),
                           'policies': shve_cases.POLICIES, 'baseline_version': BASELINE_VERSION,
                           'baseline_code_sha256': _digest(Path(shve_cases.__file__)),
                           'churn_metric_code_sha256': _digest(Path(churn.__file__)),
                           'implementation_sha256': {name:_digest(b.BASE/name) for name in (
                               'benchmark.py','shve_improved.py','shve_recovery.py','churn_data.py',
                               'churn_analysis.py','shve_cases.py','report_export.py',
                               'report_assets/report.js','report_assets/logic.js')},
                           'churn_protocol': 'Historical prior through 2025-11; calibration 2026-01/02; evaluation 2026-04/05/06/07; N/n month-label weights; supplied labels unchanged',
                           'prediction_scope': 'Retrospective supplied-label discrimination, conditional on monthly feature availability; not independently verified prospective forecasting',
                           'threshold_rule': 'Primary churn threshold 0.5 fixed before inference; supplementary maximum weighted development F1 only. Choice acceptance: >=10 development families with zero observed accepted errors, otherwise all review.',
                           'egress_requirement': 'Explicit operator approval for these prepared source features at configured hosted endpoints; never implied by synthetic policies'}
    b.validate_config(cfg)
    b.validate_data(cases, cfg)
    return cfg


def prepare_suite(folder=DEFAULT_DATA):
    folder = Path(folder)
    source = folder/'churn'
    source_manifest = b.parse((source/'manifest.json').read_text('utf-8'))
    if _digest(source/'benchmark_data.jsonl') != source_manifest['benchmark_data_sha256']:
        raise ValueError('Prepared churn evidence checksum differs.')
    cases = shve_cases.build_cases(seed=0)
    for case in cases:
        case['dataset'] = 'shve_improved_' + case['split']
    cases += [b.parse(line) for line in (source/'benchmark_data.jsonl').read_text('utf-8').splitlines() if line]
    if Counter(c['task'] for c in cases) != {**{task:60 for task in shve_cases.POLICIES}, 'churn_prediction':1000}:
        raise ValueError('Expected 360 contextual cases plus 1,000 churn cases.')
    cfg = run_config(b.default_config(), cases)
    b.validate_data(cases, cfg)
    contents = b.jsonl_bytes(cases)
    manifest = {'revision': cfg['shve_protocol']['dataset_revision'], 'cases': len(cases),
                'task_counts': dict(Counter(c['task'] for c in cases)),
                'split_counts': dict(Counter(c['split'] for c in cases)),
                'benchmark_data_sha256': hashlib.sha256(contents).hexdigest(),
                'source_churn_manifest': source_manifest,
                'semantic_policy_sha256': _digest(shve_cases.__file__),
                'semantic_labels': 'Authored experimental policy fixtures, not human-adjudicated operational labels',
                'core_evaluation': '48 cases / 24 authored premise families per category; paired views share families',
                'request_fields': 'State and typed question only; identifiers, gold, weights and label reasons are analysis-only'}
    folder.mkdir(parents=True, exist_ok=True)
    for name, payload in [('benchmark_data.jsonl', contents), ('manifest.json', b.canonical(manifest).encode()+b'\n')]:
        target = folder/name
        if target.exists() and target.read_bytes()!=payload:
            raise ValueError('Refusing to replace a different frozen improved suite.')
        target.write_bytes(payload)
    return cases, manifest


def load_suite(folder=DEFAULT_DATA):
    folder = Path(folder)
    manifest = b.parse((folder/'manifest.json').read_text('utf-8'))
    if _digest(folder/'benchmark_data.jsonl') != manifest['benchmark_data_sha256']:
        raise ValueError('Improved suite checksum differs.')
    cfg = b.default_config()
    cfg['tasks']['churn_prediction'] = {'label_semantics': 'observed_outcome'}
    cases = b.load_jsonl((folder/'benchmark_data.jsonl').read_bytes(), 'SHVE improved', cfg)
    if manifest.get('revision')!='shve-improved-v1-20261007':
        raise ValueError('Unknown improved suite revision.')
    expected = {**{task:60 for task in shve_cases.POLICIES}, 'churn_prediction':1000}
    if dict(Counter(c['task'] for c in cases))!=expected or manifest.get('task_counts')!=expected:
        raise ValueError('Improved suite task counts differ from the frozen protocol.')
    for task in TASKS:
        count = Counter(c['split'] for c in cases if c['task']==task)
        if count!=({'development':200,'evaluation':800} if task=='churn_prediction' else
                   {'development':12,'evaluation':48}):
            raise ValueError('Improved suite split counts differ: '+task)
        if any(c['dataset']!='shve_improved_'+c['split'] for c in cases if c['task']==task):
            raise ValueError('Dataset names differ from the improved partitions.')
    return cases, manifest


def baseline_records(cases, cfg):
    rows = []
    for case in cases:
        started = time.perf_counter()
        state = b.state_for(case)
        value = (state['historical_positive_rate'] if case['task']=='churn_prediction' else
                 shve_cases.baseline(case['task'], state, case['question']))
        valid, correct, numeric, decision = b.grade(value, case, cfg)
        rows.append({'model': 'rules', 'id': case['id'], 'dataset': case['dataset'], 'task': case['task'],
                     'cluster_id': case['cluster_id'], 'type': case['question']['type'],
                     'expected': case['expected'], 'value': value, 'decision': decision,
                     'valid': valid, 'correct': correct, 'numeric_error': numeric, 'api_error': None,
                     'critical': False, 'critical_error_choices': case.get('critical_error_choices', []),
                     'phase': 'primary', 'scored': True, 'latency_s': time.perf_counter()-started,
                     'latency_basis': 'in-process frozen lexical/historical-prior control, not HTTP end-to-end',
                     'resolved_model': BASELINE_VERSION, 'confidence': None, 'probabilities': None,
                     'input_tokens': 0, 'output_tokens': 0, '_api_cost': 0.0})
    return rows


def churn_summaries(records, cases, cfg):
    relevant = [c for c in cases if c['task']=='churn_prediction']
    if not relevant:
        return []
    aliases = [m['name'] for m in cfg['models'] if m.get('enabled')]
    result = []
    development = [c for c in relevant if c['split']=='development']
    for alias in aliases:
        rows = [r for r in records if r['model']==alias and r['task']=='churn_prediction' and b.primary_row(r)]
        dev_ids = {c['id'] for c in development}
        chosen = churn.choose_threshold([r for r in rows if r['id'] in dev_ids], development)
        for dataset in sorted({c['dataset'] for c in relevant}):
            cohort = [c for c in relevant if c['dataset']==dataset]
            ids = {c['id'] for c in cohort}
            observed = [r for r in rows if r['id'] in ids]
            fixed = churn.prediction_quality(observed, cohort, threshold=.5)
            selected = (churn.prediction_quality(observed, cohort, threshold=chosen['threshold'])
                        if chosen['threshold'] is not None else None)
            result.append({'model': alias, 'dataset': dataset, 'task': 'churn_prediction',
                           'fixed': fixed, 'selected': selected, 'threshold_selection': chosen,
                           'baseline': churn.prediction_quality(baseline_records(cohort, cfg), cohort, threshold=.5)})
    return result


def freeze_thresholds(records, cases, cfg):
    # The legacy algorithm is reusable but its hardcoded task list is not.
    index = {c['id']:c for c in cases}
    policies = {}
    for alias in {r['model'] for r in records}:
        policies[alias] = {}
        for task in shve_cases.POLICIES:
            dev = [r for r in records if b.primary_row(r) and r['model']==alias and
                   r['task']==task and index[r['id']]['split']=='development']
            base = {**d.policy_for(cfg,task), 'mode':'probability'}
            chosen = None
            for threshold in (0,.5,.6,.7,.8,.9,.95,.99,1):
                candidate = {**base,'threshold':threshold}
                accepted = [r for r in dev if d.acceptance(r,candidate)=='accepted']
                if len({r['cluster_id'] for r in accepted})>=10 and all(r['correct'] for r in accepted):
                    chosen = candidate
                    break
            policies[alias][task] = {**(chosen or {**base,'threshold':1,'accept_none':True}),
                                    'selection_basis':'development only; provisional; evaluation never used',
                                    'development_cases':len(dev),
                                    'development_families':len({r['cluster_id'] for r in dev}),
                                    'status': 'selected' if chosen else 'insufficient_calibration_for_ten_family_gate'}
    return policies


def recovery_config(previous, cases, current, selected):
    from shve_recovery import remaining_calls
    if previous['config'].get('shve_protocol',{}).get('suite')!='improved':
        raise ValueError('Recovery requires an improved suite source.')
    if b.input_json(sorted(cases,key=lambda c:c['id'])) != b.input_json(sorted(previous['cases'],key=lambda c:c['id'])):
        raise ValueError('Recovery data differs from the frozen source.')
    if not selected or set(selected)-set(MODELS):
        raise ValueError('Choose existing incomplete model aliases.')
    cfg = copy.deepcopy(previous['config'])
    indexed = {m['name']:m for m in current['models']}
    for model in cfg['models']:
        alias = model['name']
        model['enabled'] = alias in selected
        if alias not in selected:
            continue
        if not remaining_calls(previous,alias):
            raise ValueError('Refusing to rerun a complete model: '+alias)
        source = indexed[alias]
        if (source.get('model'),source.get('endpoint')) != (model.get('model'),model.get('endpoint')):
            raise ValueError('Configured endpoint/model differs from the frozen source.')
        for key in ('api_key','api_key_env','headers'):
            if key in source:
                model[key] = copy.deepcopy(source[key])
    return cfg


def export_analysis(folder):
    import report_export
    folder = Path(folder)
    report = b.read_report(folder)
    errors, warnings = b.audit_report(report)
    if errors or warnings:
        raise ValueError('Frozen evidence audit failed: '+str(errors+warnings))
    if report['config'].get('shve_protocol',{}).get('suite')!='improved':
        raise ValueError('This exporter requires an improved suite.')
    evidence = {name:_digest(folder/name) for name in ('raw.jsonl','data.snapshot.jsonl','config.snapshot.json','manifest.json')}
    cfg = copy.deepcopy(report['config'])
    cfg['frozen_acceptance'] = freeze_thresholds(report['records'],report['cases'],cfg)
    payload = report_export.build_payload(report,analysis_cfg=cfg)
    (folder/'report.html').write_text(report_export.render_report(payload),encoding='utf-8')
    summaries = payload['summaries_by_reference'][payload['baseline']]
    decision = payload['decision_analysis']
    b.write_json(folder/'decision.analysis.json',summaries)
    b.write_json(folder/'churn_prediction.analysis.json',decision['churn_prediction'])
    b.write_json(folder/'thresholds.analysis.json',cfg['frozen_acceptance'])
    b.write_json(folder/'decision_diagnostics.json',decision['diagnostics'])
    for name, rows in {'per_task_metrics.csv': summaries, 'baseline_summary.csv':decision['baseline_summaries'],
                       'business_decisions.csv':decision['baseline_comparisons'], 'risk_curves.csv':decision['risk_curves'],
                       'per_label.csv':[{'model':r['model'],'dataset':r['dataset'],'task':r['task'],**label}
                                        for r in summaries for label in r['per_label']],
                       'error_directions.csv':[{'model':r['model'],'dataset':r['dataset'],'task':r['task'],**direction}
                                               for r in summaries for direction in r['error_directions']]}.items():
        (folder/name).write_bytes(b.csv_bytes(rows))
    (folder/'baseline.raw.jsonl').write_bytes(b.jsonl_bytes(decision['baseline_records']))
    lines=['# Improved SHVE run — '+folder.name,'',
           'Six contextual policy categories plus retrospective supplied-label churn prediction.',
           'All original report tabs and evidence are included in report.html. No downstream specialist execution or retention uplift was measured.',
           '', '| Evaluation category | Model | Task success | 95% interval | Valid answers |',
           '|---|---|---:|---|---:|']
    for row in summaries:
        if row['evaluation_scope']!='evaluation' or row['task']=='churn_prediction':continue
        interval=(f"{row['success_ci_low']:.1%}–{row['success_ci_high']:.1%}"
                  if b.finite(row['success_ci_low']) and b.finite(row['success_ci_high']) else 'unavailable')
        success=f"{row['success_rate']:.1%}" if row['status']=='complete' else 'incomplete'
        lines.append(f"| {row['task']} | {row['model']} | {success} | {interval} | {row['valid_rate']:.1%} |")
    lines += ['', '## Population-weighted churn prediction',
              'Primary threshold is fixed at 0.5; supplementary operating points use calibration only. Scores use N/n sampling weights. Read probability metrics together with response coverage. The historical constant probability is a reference, not a trained tabular challenger.',
              '', '| Model | ROC-AUC | Average precision | Brier | Log loss | Valid probability coverage |',
              '|---|---:|---:|---:|---:|---:|']
    for entry in decision['churn_prediction']:
        if entry['fixed']['split']!='evaluation':continue
        quality=entry['fixed'];p=quality['probability_metrics']
        values=[f'{p[k]:.5f}' if b.finite(p[k]) else 'unavailable' for k in ('roc_auc','average_precision','brier','log_loss')]
        lines.append('| '+entry['model']+' | '+' | '.join(values)+f" | {quality['coverage']['weighted_probability_validity']:.1%} |")
    lines += ['', '## Interpretation',
              'Observed leaders describe this sample, with 24 authored evaluation premise families per core task. Lexical control comparisons establish an advantage over that frozen control only. Choice automation is withheld because six calibration families cannot satisfy the ten-family gate; this is not inference failure.',
              cfg['shve_protocol']['prediction_scope'],
              'Human adjudication and a fresh operational confirmation sample with stated risk/latency targets are required for deployment decisions. Prediction performance does not measure intervention uplift; monetary scenarios are assumptions, not measured savings.']
    (folder/'business_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if any(_digest(folder/name)!=value for name,value in evidence.items()):
        raise RuntimeError('Analysis modified frozen evidence.')
    b.write_json(folder/'analysis_provenance.json',{'frozen_evidence_sha256':evidence,
                 'basis':'Saved-response analysis only; no new inference, labels or source data changes',
                 'baseline_version':BASELINE_VERSION,'generated_utc':b.utc()})
    b.seal(folder)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--churn-source',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--only',nargs='+',choices=MODELS)
    parser.add_argument('--recover-from',type=Path)
    parser.add_argument('--egress-approved',action='store_true')
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    if args.report:
        export_analysis(args.report)
        return
    if args.prepare:
        if args.churn_source:
            import churn_data
            churn_data.prepare(args.churn_source,args.data/'churn')
        cases, checks = prepare_suite(args.data)
    else:
        cases, checks = load_suite(args.data)
    current = b.read_config()
    cfg = run_config(current,cases)
    selected = args.only or list(MODELS)
    if args.recover_from:
        previous = b.read_report(args.recover_from)
        errors,warnings = b.audit_report(previous)
        if errors or warnings:
            raise ValueError('Recovery source evidence audit failed.')
        cfg = recovery_config(previous,cases,current,selected)
    b.validate_data(cases,cfg)
    plan = b.request_plan(cases,cfg,selected)
    output = b.BASE/'results/shve-improved-preflight'
    output.mkdir(parents=True,exist_ok=True)
    public = {'request_plan':plan,'selected_models':selected,'data_sha256':checks['benchmark_data_sha256'],
              'credentials_available':{m['name']:bool(b.model_key(m)) for m in cfg['models'] if not b.local_model(m)},
              'hosted_egress_operator_declaration':bool(args.egress_approved),
              'recovery_source':str(args.recover_from) if args.recover_from else None}
    b.write_json(output/'data_checks.json',checks)
    b.write_json(output/'request_plan.json',public)
    print(json.dumps(public,indent=2),flush=True)
    if not args.execute:
        return
    hosted = [m for m in cfg['models'] if m['name'] in selected and not b.local_model(m)]
    if hosted and not args.egress_approved:
        raise ValueError('New source feature egress must be approved before hosted calls.')
    if any(not b.model_key(m) for m in hosted):
        raise ValueError('Configure keys for all selected hosted models before executing.')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import local_runtime
    cancel = threading.Event()
    @contextmanager
    def session(model):
        with local_runtime.model_session(model,cancel):
            shve.capture_identity(model)
            yield
    def progress(event):
        if event.get('row') and event['issued']%50==0:
            print(f"{event['issued']}/{event['maximum']}",flush=True)
        elif not event.get('row') and event.get('message') and ' · ' not in event['message']:
            print(event['message'],flush=True)
    if args.recover_from:
        from shve_recovery import recover
        result = recover(previous,cfg,selected,session,cancel,progress)
    else:
        result = b.run(cfg,cases,model_session=session,cancel=cancel,progress=progress,selected=selected)
    folder = Path(result['folder'])
    for name in ('data_checks.json','request_plan.json'):
        (folder/name).write_bytes((output/name).read_bytes())
    export_analysis(folder)
    b.write_json(output/'completed_run.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
