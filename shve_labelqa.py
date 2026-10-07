"""Correct only the missing label-QA dictionary, preserving prior run evidence."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import threading

import benchmark as b
import shve
import shve_cases
import shve_improved as improved
import shve_recovery as recovery

REVISION = 'label-qa-sector-dictionary-v2-20261007'


def checked(report, complete=True):
    errors, warnings = b.audit_report(report)
    if errors or warnings:
        raise ValueError('Source evidence audit failed: '+str(errors+warnings))
    aliases = [m['name'] for m in report['config']['models'] if m.get('enabled')]
    if report['config'].get('shve_protocol', {}).get('suite')!='improved' or set(aliases)!=set(improved.MODELS):
        raise ValueError('An improved four-model source is required.')
    if complete and any(recovery.remaining_calls(report,a) for a in aliases):
        raise ValueError('Source protocol is incomplete.')


def corrected_cases(source):
    cases=copy.deepcopy([c for c in source['cases'] if c['task']=='label_qa'])
    if len(cases)!=60:
        raise ValueError('Expected exactly 60 frozen label_qa cases.')
    for case in cases:
        case['question']['instructions']=shve_cases.POLICIES['label_qa']
        case['provenance']['policy_version']=REVISION
    return cases


def provenance(report, task):
    folder=Path(report['folder'])
    return {'folder':str(folder.resolve()),'task_scope':task,
            'evidence_sha256':hashlib.sha256((folder/'evidence.json').read_bytes()).hexdigest()}


def saved_plan(report):
    path=Path(report['folder'])/'run.plan.json'
    if path.exists():return b.parse(path.read_text())
    batches={}
    for row in report['records']:
        if not b.primary_row(row):continue
        if row['id'] in batches and batches[row['id']]!=row['batch']:
            raise ValueError('Source models disagree on batch assignments.')
        batches[row['id']]=row['batch']
    if set(batches)!={c['id'] for c in report['cases']}:
        raise ValueError('Source batch assignments are incomplete.')
    return {'protocol':report['manifest']['protocol'],'batches':batches,
            'stability_sample_ids':report['manifest']['stability_sample_ids'],
            'request_serialization':report['manifest'].get('request_serialization','sorted_keys')}


def validate_correction(source, corrected):
    errors,warnings=b.audit_report(corrected)
    if errors or warnings:raise ValueError('Correction evidence audit failed.')
    if b.input_json(sorted(corrected['cases'],key=lambda c:c['id']))!=b.input_json(sorted(corrected_cases(source),key=lambda c:c['id'])):
        raise ValueError('Correction changed evidence, labels or unrelated questions.')
    if corrected['config']['shve_protocol'].get('category_revision_source')!=provenance(source,'label_qa'):
        raise ValueError('Correction source provenance differs.')
    source_models={m['name']:m for m in source['config']['models']}
    if {m['name'] for m in corrected['config']['models']}!=set(source_models):
        raise ValueError('Correction model participants differ.')
    for model in corrected['config']['models']:
        for key in ('api','model','endpoint','params','structured_output'):
            if model.get(key)!=source_models[model['name']].get(key):
                raise ValueError('Correction changed the model configuration: '+model['name'])
    ids={c['id'] for c in corrected['cases']}
    repeats=[i for i in source['manifest']['stability_sample_ids'] if i in ids]
    protocol=copy.deepcopy(source['manifest']['protocol'])
    protocol.update(warmup_calls=2,repeat_cases=len(repeats),
                    max_requests=(60+len(repeats)*(protocol['repetitions']-1)+2)*4)
    if corrected['manifest']['protocol']!=protocol or corrected['manifest']['stability_sample_ids']!=repeats:
        raise ValueError('Correction changed protocol or repeat sample.')
    original_plan=saved_plan(source)
    revised_plan=saved_plan(corrected)
    if any(revised_plan['batches'].get(i)!=original_plan['batches'].get(i) for i in ids):
        raise ValueError('Correction changed batch assignments.')


def execution_config(source, seed, current):
    checked(source)
    validate_correction(source,seed)
    selected=[a for a in improved.MODELS if recovery.remaining_calls(seed,a)]
    if not selected:return copy.deepcopy(seed['config']),[]
    return improved.recovery_config(seed,seed['cases'],current,selected),selected


def prepare_seed(source, output_root=None):
    checked(source)
    cases=corrected_cases(source)
    old={c['id']:c for c in source['cases']}
    if all(c['question']['instructions']==old[c['id']]['question']['instructions'] for c in cases):
        raise ValueError('This source already contains the corrected dictionary.')
    ids={c['id'] for c in cases}
    repeats=[i for i in source['manifest']['stability_sample_ids'] if i in ids]
    batches={i:v for i,v in saved_plan(source)['batches'].items() if i in ids}
    cfg=copy.deepcopy(source['config'])
    cfg['protocol'].update(warmup_calls=2,repeat_cases=len(repeats))
    cfg['protocol']['max_requests']=(len(cases)+len(repeats)*(cfg['protocol']['repetitions']-1)+2)*4
    cfg['shve_protocol'].update(category_revision=REVISION,category_revision_source=provenance(source,'label_qa'),
                              category_revision_code_sha256={name:hashlib.sha256((b.BASE/name).read_bytes()).hexdigest()
                                  for name in ('shve_cases.py','shve_labelqa.py','benchmark.py','shve_recovery.py')},
                              input_sha256=b.fingerprint(sorted(cases,key=lambda c:c['id'])))
    cfg['shve_protocol']['policies']['label_qa']=shve_cases.POLICIES['label_qa']
    cfg['tasks']['label_qa'].pop('question',None)
    b.validate_data(cases,cfg)
    root=Path(output_root or b.BASE/'results')
    with b.run_lock(root):
        folder=root/(b.datetime.now(b.timezone.utc).strftime('%Y%m%d_%H%M%S')+'_labelqa_seed_'+os.urandom(3).hex())
        folder.mkdir()
        plan=b.request_plan(cases,cfg)
        b.write_json(folder/'run.plan.json',{'batches':batches,'stability_sample_ids':repeats,
                                          'protocol':cfg['protocol'],'request_plan':plan,'request_serialization':'preserve_order'})
        b.write_json(folder/'category_revision.json',{'revision':REVISION,'source':provenance(source,'label_qa'),
                     'change':'Sector dictionary added to the question; evidence, gold, splits, batches and repeat IDs preserved.'})
        b.save_report(folder,[],cases,cfg,{},repeat_ids=repeats,batches=batches,protocol=cfg['protocol'],
                      plan=plan,models=cfg['models'],started=b.utc(),execution_order='pending_category_revision')
    return folder


def revise_report(source, corrected, output_root=None):
    checked(source); checked(corrected)
    validate_correction(source,corrected)
    expected=sorted(corrected_cases(source),key=lambda c:c['id'])
    if b.input_json(sorted(corrected['cases'],key=lambda c:c['id']))!=b.input_json(expected):
        raise ValueError('Correction changed evidence, labels or unrelated questions.')
    source_models={m['name']:m for m in source['config']['models']}
    for model in corrected['config']['models']:
        original=source_models[model['name']]
        for key in ('api','model','endpoint','params','structured_output'):
            if model.get(key)!=original.get(key):
                raise ValueError('Correction changed the model configuration: '+model['name'])
    ids={c['id'] for c in expected}
    repeats=[i for i in source['manifest']['stability_sample_ids'] if i in ids]
    if corrected['manifest']['stability_sample_ids']!=repeats:
        raise ValueError('Correction changed the repeat sample.')
    original_plan=saved_plan(source)
    revised_plan=saved_plan(corrected)
    if any(revised_plan['batches'].get(i)!=original_plan['batches'].get(i) for i in ids):
        raise ValueError('Correction changed batch assignments.')
    cfg=copy.deepcopy(source['config'])
    cfg['protocol']['warmup_calls']+=corrected['manifest']['protocol']['warmup_calls']
    cfg['protocol']['max_requests']+=8
    cfg['shve_protocol']['category_revision']=REVISION
    cfg['shve_protocol']['category_revision_source']=provenance(source,'label_qa')
    cfg['shve_protocol']['category_revision_code_sha256']=corrected['config']['shve_protocol']['category_revision_code_sha256']
    cfg['shve_protocol']['policies']['label_qa']=shve_cases.POLICIES['label_qa']
    cfg['tasks']['label_qa'].pop('question',None)
    replacement={c['id']:c for c in expected}
    cases=[copy.deepcopy(replacement.get(c['id'],c)) for c in source['cases']]
    cfg['shve_protocol']['input_sha256']=b.fingerprint(sorted(cases,key=lambda c:c['id']))
    b.validate_data(cases,cfg)
    records=[]
    for report, rows in ((source,[r for r in source['records'] if r['task']!='label_qa']),
                         (corrected,corrected['records'])):
        for saved in rows:
            row=copy.deepcopy(saved)
            row.update(source_run=Path(report['folder']).name,source_request_index=saved['request_index'],
                       request_index=len(records)+1)
            records.append(row)
    warmups={a:[r['id'] for r in records if r['model']==a and r['phase']=='warmup'] for a in improved.MODELS}
    if len({tuple(v) for v in warmups.values()})!=1:
        raise ValueError('Model warm-up schedules differ.')
    warmup_ids=warmups[improved.MODELS[0]]
    root=Path(output_root or b.BASE/'results')
    with b.run_lock(root):
        folder=root/(b.datetime.now(b.timezone.utc).strftime('%Y%m%d_%H%M%S')+'_labelqa_revised_'+os.urandom(3).hex())
        folder.mkdir()
        plan=b.request_plan(cases,cfg)
        sources=[provenance(source,'all except label_qa'),provenance(corrected,'label_qa plus its session warm-ups')]
        b.write_json(folder/'category_revision.json',{'revision':REVISION,'sources':sources,
            'superseded_requests':sum(r['task']=='label_qa' for r in source['records']),
            'replacement_requests':len(corrected['records']),
            'policy':'Explicit category supersession. Original reports untouched; other answers reused exactly. Both sessions warm-ups retained.'})
        b.write_json(folder/'run.plan.json',{**original_plan,'protocol':cfg['protocol'],'request_plan':plan,'warmup_case_ids':warmup_ids})
        b.save_report(folder,records,cases,cfg,source['availability'],repeat_ids=source['manifest']['stability_sample_ids'],
                      batches=original_plan['batches'],protocol=cfg['protocol'],plan=plan,models=cfg['models'],
                      started=source['manifest']['started_utc'],execution_order='explicit_category_revision',sources=sources,
                      warmup_case_ids=warmup_ids)
    return folder


def merge_progress(previous, executed, selected, output_root=None):
    """Keep complete aliases and every partial attempt after repeated interruption."""
    for report in (previous,executed):
        if b.audit_report(report)!=([],[]):raise ValueError('Progress source audit failed.')
    if (b.input_json(previous['cases'])!=b.input_json(executed['cases']) or
            b.canonical({k:v for k,v in previous['config'].items() if k!='models'})!=
            b.canonical({k:v for k,v in executed['config'].items() if k!='models'})):
        raise ValueError('Progress sources differ in data or scoring configuration.')
    cfg=copy.deepcopy(previous['config'])
    by_model={m['name']:m for m in executed['config']['models']}
    for model in cfg['models']:
        if model['name'] in selected:model.update(copy.deepcopy(by_model[model['name']]))
        model['enabled']=True
    records=[]
    for report, rows in ((previous,[r for r in previous['records'] if r['model'] not in selected]),
                         (executed,executed['records'])):
        for saved in rows:
            row=copy.deepcopy(saved)
            row.update(source_run=Path(report['folder']).name,source_request_index=row['request_index'],request_index=len(records)+1)
            records.append(row)
    plan=saved_plan(previous)
    merged={'cases':previous['cases'],'records':records,'manifest':previous['manifest']}
    for alias in improved.MODELS:recovery.remaining_calls(merged,alias)
    root=Path(output_root or b.BASE/'results')
    with b.run_lock(root):
        folder=root/(b.datetime.now(b.timezone.utc).strftime('%Y%m%d_%H%M%S')+'_labelqa_progress_'+os.urandom(3).hex())
        folder.mkdir()
        b.write_json(folder/'run.plan.json',plan)
        b.save_report(folder,records,previous['cases'],cfg,{**previous['availability'],**executed['availability']},
                      repeat_ids=plan['stability_sample_ids'],batches=plan['batches'],protocol=plan['protocol'],
                      plan=b.request_plan(previous['cases'],cfg),models=cfg['models'],started=previous['manifest']['started_utc'],
                      interrupted=executed['manifest']['interrupted'],error=executed['manifest']['error'],
                      execution_order='category_recovery_progress',sources=[provenance(previous,'previous correction attempts'),
                                                                           provenance(executed,'remaining correction attempts')])
    return folder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--corrected',type=Path)
    args=parser.parse_args()
    source=b.read_report(args.source)
    if args.corrected:
        folder=revise_report(source,b.read_report(args.corrected))
        improved.export_analysis(folder)
        print(json.dumps({'revised_report':str(folder)}),flush=True)
        return
    seed=b.read_report(args.resume or prepare_seed(source))
    cfg,selected=execution_config(source,seed,b.read_config())
    hosted=[m for m in cfg['models'] if m['name'] in selected and not b.local_model(m)]
    if any(not b.model_key(m) for m in hosted):raise ValueError('Configured keys are missing.')
    print(json.dumps({'seed':str(seed['folder']),'plan':b.request_plan(seed['cases'],cfg)}),flush=True)
    if not args.execute:return
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import local_runtime
    cancel=threading.Event()
    @contextmanager
    def session(model):
        with local_runtime.model_session(model,cancel):
            shve.capture_identity(model)
            yield
    def progress(event):
        if event.get('message') or event['issued']%20==0:
            print(json.dumps({k:v for k,v in event.items() if k!='row'}),flush=True)
    if selected:
        result=recovery.recover(seed,cfg,selected,session,cancel,progress)
        complete=[a for a in improved.MODELS if a not in selected]
        if complete:
            result['folder']=str(merge_progress(seed,b.read_report(result['folder']),selected))
    else:
        result={'folder':str(seed['folder']),'error':None,'interrupted':False}
    print(json.dumps({'corrected_run':result}),flush=True)
    folder=revise_report(source,b.read_report(result['folder']))
    improved.export_analysis(folder)
    output=b.BASE/'results/shve-labelqa-preflight';output.mkdir(exist_ok=True)
    b.write_json(output/'completed_run.json',{'corrected_run':result['folder'],'revised_report':str(folder)})
    print(json.dumps({'revised_report':str(folder)}),flush=True)


if __name__=='__main__':main()
