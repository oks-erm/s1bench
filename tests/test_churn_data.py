import json
import tempfile
import unittest
from pathlib import Path

import benchmark as b


class ChurnDataTests(unittest.TestCase):
    def test_sample_hides_identifiers_outcomes_and_future_flags(self):
        import churn_data as c
        row={'ID_anon':'PRIVATE_CUSTOMER','ChurnNextMonth':1,'snapshot_month':'2026-04-01',
             'tenure_months':18,'delivery_count':0,'cases_rows_in_month':None,
             'negative_ratio':None,'contract_end_flag':1,'contract_age_days':-4}
        item=c.make_case(row, population_count=500, sample_count=50, prior=.005,
                         source_sha='source-fingerprint')
        self.assertIsInstance(item,dict)
        self.assertTrue(item['expected'])
        self.assertEqual(item['metadata']['sampling_weight'],10)
        state=json.dumps(b.state_for(item))
        for value in ['PRIVATE_CUSTOMER','ChurnNextMonth','contract_end_flag','contract_age_days']:
            self.assertNotIn(value,state)
        self.assertEqual(item['input']['snapshot']['cases_rows_in_month'],0)
        self.assertIsNone(item['input']['snapshot']['negative_ratio'])
        self.assertEqual(item['split'],'evaluation')

    def test_invalid_labels_and_quarantined_months_cannot_be_scored(self):
        import churn_data as c
        for label,month in [(None,'2026-04-01'),(2,'2026-04-01'),(0,'2026-09-01')]:
            with self.assertRaises(ValueError):
                c.make_case({'ID_anon':'a','ChurnNextMonth':label,'snapshot_month':month},
                            population_count=100,sample_count=10,prior=.005,source_sha='x')

    def test_stratified_sampling_weights_recover_actual_population(self):
        import churn_data as c
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows=[{'ID_anon':'train','snapshot_month':'2025-11-01','ChurnNextMonth':1}]
        for month in ['2026-01-01','2026-04-01']:
            for i in range(6):
                rows.append({'ID_anon':month+str(i),'snapshot_month':month,'ChurnNextMonth':int(i==0)})
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'source.parquet';dest=Path(tmp)/'prepared'
            pq.write_table(pa.Table.from_pylist(rows),src)
            quotas={'2026-01':{0:2,1:1},'2026-04':{0:2,1:1}}
            result=c.prepare(src,dest,quotas=quotas,seed=19)
            cfg=b.default_config();cfg['tasks']['churn_prediction']={'label_semantics':'observed_outcome'}
            cases=b.load_data(dest/'benchmark_data.jsonl',cfg)
            self.assertEqual(len(cases),6)
            self.assertEqual(sum(x['metadata']['sampling_weight'] for x in cases),12)
            self.assertEqual(sum(x['metadata']['sampling_weight'] for x in cases if x['expected']),2)
            self.assertEqual(result['population_counts']['2026-04'],{'0':5,'1':1})
            original=(dest/'benchmark_data.jsonl').read_bytes()
            c.prepare(src,Path(tmp)/'again',quotas=quotas,seed=19)
            self.assertEqual(original,(Path(tmp)/'again/benchmark_data.jsonl').read_bytes())


if __name__=='__main__':unittest.main()
