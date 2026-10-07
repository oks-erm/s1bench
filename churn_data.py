"""Prepare a frozen, weighted monthly churn sample without exposing identifiers."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path
import random

import benchmark as b

FEATURES=('tenure_months','delivery_count','days_since_last_delivery','invoice_count',
          'days_since_last_invoice','price_changes_3m','months_since_price_change',
          'price_increase_flag','price_decrease_flag','price_vs_district_avg',
          'cases_rows_in_month','complaints_3m','billing_complaints','delivery_complaints',
          'runout_complaints','price_complaints','negative_ratio','sentiment_3m','Sector')
EVENT_COUNTS={'cases_rows_in_month','billing_complaints','delivery_complaints',
              'runout_complaints','price_complaints'}
QUOTAS={**{m:{0:50,1:50} for m in ('2026-01','2026-02')},
        **{m:{0:150,1:50} for m in ('2026-04','2026-05','2026-06','2026-07')}}


def _month(value):
    text=str(value)[:7]
    try:
        year,month=map(int,text.split('-'))
    except (ValueError,TypeError):
        raise ValueError('Invalid snapshot month') from None
    if len(text)!=7 or not 1<=month<=12 or not 2000<=year<=2100:
        raise ValueError('Invalid snapshot month')
    return text


def _number(value):
    if value is None:return None
    if isinstance(value,bool):return int(value)
    try:result=float(value)
    except (ValueError,TypeError):return None
    return result if math.isfinite(result) else None


def make_case(row, *, population_count, sample_count, prior, source_sha):
    month=_month(row.get('snapshot_month'))
    if month not in QUOTAS:raise ValueError('Month is outside the frozen churn protocol')
    label=row.get('ChurnNextMonth')
    if label not in (0,1) or label is None:raise ValueError('Missing or invalid supplied churn label')
    identity=row.get('ID_anon')
    if identity is None or not str(identity).strip():raise ValueError('Missing grouping ID')
    if not (0<sample_count<=population_count) or not 0<=prior<=1:
        raise ValueError('Invalid sampling counts or historical prior')
    customer=hashlib.sha256(str(identity).encode()).hexdigest()
    fingerprint=hashlib.sha256((customer+':'+month).encode()).hexdigest()[:24]
    snapshot={}
    for key in FEATURES:
        if key not in row:continue
        snapshot[key]=str(row[key]) if key=='Sector' and row[key] is not None else _number(row[key])
        if key in EVENT_COUNTS and row[key] is None:snapshot[key]=0
    split='development' if month in ('2026-01','2026-02') else 'evaluation'
    return {'id':'churn:'+fingerprint,'task':'churn_prediction',
            'dataset':'shve_improved_'+split,'dataset_role':split,'split':split,
            'cluster_id':'customer:'+customer,'expected':bool(label),
            'input':{'snapshot':snapshot,'snapshot_month':month,
                     'historical_positive_rate':prior,
                     'context':'Monthly customer/site aggregates. The support extract is complete; null monthly support counts mean no events, while sentiment and event ratios are not applicable with no events. Missing delivery recency means no known prior delivery. Numeric quantities are used as supplied; no unit conversion or causal interpretation is authorised.'},
            'question':{'type':'noul','instructions':
                'Estimate the probability that this customer/site receives the supplied next-month churn outcome. '
                'Predict from this snapshot and the historical positive rate; do not classify missing support sentiment as a complaint. '
                'The target is ChurnNextMonth as supplied by the dataset, not a proven cancellation cause. '
                'Return a probability between zero and one. There are no future closure flags or outcome labels in the snapshot.'},
            'provenance':'supplied_anonymised_monthly_snapshots',
            'label_source':'supplied_ChurnNextMonth_unchanged','review_status':'observed_supplied_label',
            'metadata':{'sampling_weight':population_count/sample_count,
                        'sampling_stratum':month+':'+str(int(label)),
                        'population_stratum_count':population_count,'sample_stratum_count':sample_count,
                        'source_sha256':source_sha,'unit_conversion':'none'}}


def prepare(source, destination, *, quotas=None, seed=20261007):
    """Reservoir-sample customer-months uniformly within month/label strata."""
    import pyarrow.parquet as pq  # Optional data-preparation dependency, not inference.
    source=Path(source);destination=Path(destination)
    quotas=quotas or QUOTAS
    if any(m not in QUOTAS or set(q)!={0,1} or any(not isinstance(n,int) or n<=0 for n in q.values())
           for m,q in quotas.items()):raise ValueError('Invalid sampling quotas')
    digest=hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    source_sha=digest.hexdigest()
    parquet=pq.ParquetFile(source)
    required={'ID_anon','snapshot_month','ChurnNextMonth'}
    if not required.issubset(parquet.schema_arrow.names):raise ValueError('Required churn source fields missing')
    columns=[c for c in (*required,*FEATURES) if c in parquet.schema_arrow.names]
    counts=defaultdict(Counter);reservoirs=defaultdict(list)
    generators={(m,y):random.Random(f'{seed}:{m}:{y}') for m,q in quotas.items() for y in q}
    training=Counter();seen=0
    for batch in parquet.iter_batches(batch_size=65536,columns=columns):
        data=batch.to_pydict()
        for i,(date,label) in enumerate(zip(data['snapshot_month'],data['ChurnNextMonth'])):
            month=_month(date)
            if label not in (0,1) or label is None:raise ValueError('Source contains invalid churn outcomes')
            if month<='2025-11':training[int(label)]+=1
            if month not in quotas:continue
            key=(month,int(label));counts[month][int(label)]+=1;seen+=1
            priority=generators[key].random();heap=reservoirs[key];quota=quotas[month][int(label)]
            if len(heap)<quota or priority < -heap[0][0]:
                row={c:values[i] for c,values in data.items()}
                entry=(-priority,seen,row)
                if len(heap)<quota:heapq.heappush(heap,entry)
                else:heapq.heapreplace(heap,entry)
    if not sum(training.values()):raise ValueError('No historical rows for the frozen prior')
    prior=training[1]/sum(training.values())
    cases=[]
    for month,q in sorted(quotas.items()):
        for label,quota in sorted(q.items()):
            selected=reservoirs[month,label]
            if len(selected)!=quota:raise ValueError(f'Insufficient source rows for stratum {month}:{label}')
            cases.extend(make_case(row,population_count=counts[month][label],sample_count=quota,
                                   prior=prior,source_sha=source_sha) for _,_,row in selected)
    cases.sort(key=lambda c:c['id'])
    cfg=b.default_config();cfg['tasks']['churn_prediction']={'label_semantics':'observed_outcome'}
    b.validate_data(cases,cfg)
    contents=b.jsonl_bytes(cases)
    manifest={'revision':'churn-snapshot-v1-20261007','source_sha256':source_sha,
              'benchmark_data_sha256':hashlib.sha256(contents).hexdigest(),'seed':seed,'cases':len(cases),
              'population_counts':{m:{str(y):n for y,n in sorted(c.items())} for m,c in sorted(counts.items())},
              'historical_prior':prior,'historical_counts':dict(training),'prior_last_month':'2025-11',
              'calibration_months':['2026-01','2026-02'],'evaluation_months':['2026-04','2026-05','2026-06','2026-07'],
              'boundary_gap_months':['2025-12','2026-03'],'quarantined_months':['2026-08','2026-09'],
              'sampling':'Uniform reservoir without replacement within month/label; inclusion weight N/n',
              'features':list(FEATURES),'excluded_fields':['ID_anon','ChurnNextMonth','contract_end_flag','contract_start_flag','contract_ends_within_3m','contract_age_days'],
              'support_completeness':'User confirmed complete monthly extract; null monthly counts are zero; sentiment/ratios remain NA',
              'outcome_definition':'Supplied ChurnNextMonth unchanged; generating query unavailable',
              'feature_availability':'Source monthly aggregates assumed available at snapshot; no independent as-of join verification',
              'estimand':'Customer/site-month outcomes in these supplied source partitions; not retention uplift',
              'status':'prepared_not_run'}
    destination.mkdir(parents=True,exist_ok=True)
    for name,payload in [('benchmark_data.jsonl',contents),('manifest.json',b.canonical(manifest).encode()+b'\n')]:
        target=destination/name
        if target.exists() and target.read_bytes()!=payload:raise ValueError('Refusing to replace different prepared churn data')
        target.write_bytes(payload)
    return manifest
