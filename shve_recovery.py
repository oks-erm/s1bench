"""Resume only missing slots of a frozen SHVE protocol, retaining prior attempts."""
from collections import Counter
import copy
import hashlib
import os
from pathlib import Path

import benchmark as b


def normalized_phase(row):
    # benchmark.run relabels a failed warm-up as connection_check. It remains an
    # already issued protocol attempt, not a license to repeat the paid request.
    return 'warmup' if row.get('phase')=='connection_check' else row.get('phase')


def remaining_calls(report, alias):
    cases = report['cases']
    by_id = {c['id']:c for c in cases}
    protocol = report['manifest']['protocol']
    warmups = report['manifest'].get('warmup_case_ids')
    if warmups is None:
        warmups = [cases[i%len(cases)]['id'] for i in range(protocol['warmup_calls'])]
    if len(warmups)!=protocol['warmup_calls'] or any(i not in by_id for i in warmups):
        raise ValueError('Invalid saved warm-up schedule.')
    scheduled = [(by_id[i],'warmup',1) for i in warmups]
    scheduled += [(c,'primary',1) for c in cases]
    scheduled += [(by_id[identifier],'repeat',repetition)
                  for repetition in range(2,protocol['repetitions']+1)
                  for identifier in report['manifest']['stability_sample_ids']]
    issued = Counter((r['id'],normalized_phase(r),r.get('repetition',1))
                     for r in report['records'] if r['model']==alias)
    expected = Counter((c['id'],phase,repetition) for c,phase,repetition in scheduled)
    if any(count>expected[slot] for slot,count in issued.items()):
        raise ValueError('Duplicate or unplanned protocol record: '+alias)
    missing=[]
    for case,phase,repetition in scheduled:
        slot=(case['id'],phase,repetition)
        if issued[slot]:
            issued[slot]-=1
        else:
            missing.append((case,phase,repetition))
    return missing


def recover(previous, cfg, selected, session, cancel, progress=None):
    """Write a new audited source without overwriting or retrying prior attempts."""
    root=b.BASE/'results'
    plans={alias:remaining_calls(previous,alias) for alias in selected}
    source=Path(previous['folder'])
    original_plan=b.parse((source/'run.plan.json').read_text('utf-8'))
    batches=original_plan['batches']
    repeats=previous['manifest']['stability_sample_ids']
    cases=copy.deepcopy(previous['cases'])
    cfg=copy.deepcopy(cfg)
    models=[m for m in cfg['models'] if m['name'] in selected]
    records=[]
    for saved in previous['records']:
        if saved['model'] not in selected:
            continue
        row=copy.deepcopy(saved)
        row.update(source_run=source.name,source_request_index=row.get('request_index'),
                   request_index=len(records)+1)
        phase=normalized_phase(row)
        if phase!=row['phase']:
            row['source_phase']=row['phase']
            row['phase']=phase
        records.append(row)
    availability={}
    plan=b.request_plan(cases,cfg,selected)
    error=None
    with b.run_lock(root):
        folder=root/(b.datetime.now(b.timezone.utc).strftime('%Y%m%d_%H%M%S')+'_recovery_'+os.urandom(3).hex())
        folder.mkdir()
        b.write_json(folder/'config.initial.json',b.scrub_config(cfg))
        (folder/'data.snapshot.jsonl').write_bytes(b.jsonl_bytes(cases))
        b.write_json(folder/'run.plan.json',{**original_plan,'request_plan':plan})
        (folder/'raw.jsonl').write_bytes(b.jsonl_bytes(records) if records else b'')
        b.write_json(folder/'recovery.plan.json',{'source':str(source.resolve()),
                    'source_evidence_sha256':hashlib.sha256((source/'evidence.json').read_bytes()).hexdigest(),
                    'preserved_requests':len(records),'remaining_requests':{a:len(p) for a,p in plans.items()},
                    'policy':'Never retry an existing slot, including failed requests; normalize failed warm-ups with source_phase retained'})
        def notify(message=None,row=None):
            if progress:
                progress({'issued':len(records),'maximum':plan['maximum_requests'],
                          'folder':str(folder),'message':message,'row':row})
        try:
            for model in models:
                alias=model['name']
                if cancel.is_set():break
                notify('Preparing '+alias+' recovery')
                consecutive=0
                try:
                    with session(model):
                        availability[alias]={'status':'ready','detail':'Recovered missing protocol slots only.'}
                        for case,phase,repetition in plans[alias]:
                            if cancel.is_set():break
                            row=b.call_model(model,case,cfg)
                            row.update(phase=phase,repetition=repetition,scored=phase=='primary',
                                       batch=batches[case['id']],request_index=len(records)+1)
                            records.append(row)
                            with (folder/'raw.jsonl').open('a',encoding='utf-8') as stream:
                                stream.write(b.input_json(row)+'\n')
                            notify(row=row)
                            consecutive=consecutive+1 if row.get('api_error') else 0
                            if consecutive>=cfg.get('max_consecutive_errors',3):
                                availability[alias]={'status':'stopped','detail':'Consecutive errors; remaining slots preserved for later recovery.'}
                                break
                except (OSError,RuntimeError,ValueError) as exc:
                    availability[alias]={'status':'unavailable','detail':str(exc)}
                    notify(alias+': '+str(exc))
        except KeyboardInterrupt:
            cancel.set()
        except Exception as exc:
            error=str(exc)
        finally:
            b.save_report(folder,records,cases,cfg,availability,repeat_ids=repeats,batches=batches,
                          protocol=previous['manifest']['protocol'],plan=plan,models=models,
                          started=previous['manifest']['started_utc'],interrupted=cancel.is_set(),error=error,
                          execution_order='model_by_model_remaining_slots',sources=[{
                              'folder':str(source.resolve()),'models':list(selected),
                              'evidence_sha256':hashlib.sha256((source/'evidence.json').read_bytes()).hexdigest(),
                              'recovery':'Preserved original attempts; issued only missing slots'}])
    return {'folder':str(folder),'error':error,'interrupted':cancel.is_set()}
