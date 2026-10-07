import copy
import unittest

import benchmark as b


class ObservedOutcomeTests(unittest.TestCase):
    def cases(self):
        first={'id':'a','task':'prediction','input':{'feature':0},
               'question':{'type':'noul','instructions':'Estimate observed outcome'},
               'expected':False,'cluster_id':'customer-a'}
        second=copy.deepcopy(first)
        second.update(id='b',expected=True,cluster_id='customer-b')
        return [first,second]

    def test_stochastic_binary_outcomes_preserve_distinct_customers(self):
        cfg=b.default_config()
        cfg['tasks']['prediction']={'label_semantics':'observed_outcome'}
        cases=self.cases()
        b.validate_data(cases,cfg)
        self.assertEqual([c['cluster_id'] for c in cases],['customer-a','customer-b'])

    def test_default_policy_guard_still_rejects_conflicting_labels(self):
        with self.assertRaisesRegex(ValueError,'contradictory gold'):
            b.validate_data(self.cases(),b.default_config())

    def test_observed_semantics_cannot_bypass_choice_gold_conflicts(self):
        cases=self.cases()
        for c in cases:
            c['question'].update(type='choice',criteria={'yes':'Yes','no':'No'})
            c['expected']='yes' if c['expected'] else 'no'
        cfg=b.default_config();cfg['tasks']['prediction']={'label_semantics':'observed_outcome'}
        with self.assertRaisesRegex(ValueError,'binary probability'):
            b.validate_data(cases,cfg)
