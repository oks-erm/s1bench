"""Contract tests: native loading, family holdout, and honest heuristic limits."""
import copy
import importlib.util
import json
import unittest
from collections import Counter, defaultdict

import benchmark


class SemanticCasesTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('shve_cases'),
                             'The semantic authored-fixture builder must exist')
        import shve_cases
        return shve_cases

    def test_native_loader_accepts_all_six_categories_without_duplicate_inputs(self):
        cases = self.module().build_cases()
        counts = Counter(c['task'] for c in cases)
        self.assertEqual(set(counts), {'sector_imputation', 'semantic_validation', 'label_qa',
                                     'entity_match', 'model_routing', 'data_gap_identification'})
        self.assertEqual(set(counts.values()), {60})
        cfg = {'tasks': {}}
        loaded = benchmark.load_jsonl('\n'.join(json.dumps(c) for c in cases), 'semantic', cfg)
        self.assertEqual(len(loaded), 360)
        self.assertEqual(benchmark.validate_data(loaded, cfg), [])

    def test_family_pairs_keep_gold_and_split_but_change_transmitted_representation(self):
        families = defaultdict(list)
        for case in self.module().build_cases():
            families[case['cluster_id']].append(case)
        self.assertEqual(len(families), 180)
        for family in families.values():
            self.assertEqual(len(family), 2)
            a, b = family
            self.assertEqual(a['expected'], b['expected'])
            self.assertEqual(a['split'], b['split'])
            self.assertEqual(a['template_group'], b['template_group'])
            self.assertNotEqual(benchmark.state_for(a), benchmark.state_for(b))
        for task in {c['task'] for c in self.module().build_cases()}:
            groups = defaultdict(set)
            for case in self.module().build_cases():
                if case['task'] == task:
                    groups[case['split']].add(case['template_group'])
            self.assertFalse(groups['development'] & groups['evaluation'])

    def test_gold_and_identifiers_never_enter_transmitted_state(self):
        banned = {'expected', 'gold_reason', 'review_status', 'template_group', 'id',
                  'cluster_id', 'pair_id', 'source_id', 'record_id'}
        def keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)
        for case in self.module().build_cases():
            self.assertFalse(banned & set(keys(case['input'])))
            self.assertNotIn(case['cluster_id'], json.dumps(case['input']))
            self.assertEqual(case['review_status'], 'authored_policy_fixture')
            self.assertEqual(case['provenance']['kind'], 'authored_policy_fixture')
            self.assertNotIn(case['expected'], case['critical_error_choices'])

    def test_label_audit_requests_explain_every_assigned_sector_code(self):
        # Catch a request that gives the activity definitions but omits the
        # numeric code meanings needed to audit the existing label.
        meanings = {'Sector 1': 'Industry', 'Sector 2': 'Domestic',
                    'Sector 3': 'Hospitality/Catering', 'Sector 4': 'Agriculture',
                    'Sector 5': 'Government', 'Sector 6': 'Transport',
                    'Sector 7': 'Aerosol', 'Sector 8': 'Other',
                    'Sector 9': 'Partner / Reseller', 'Sector 10': 'Not applicable'}
        for case in self.module().build_cases():
            if case['task'] != 'label_qa':
                continue
            request = json.loads(benchmark.chat_messages(case)[1]['content'])
            instructions = request['question']['instructions']
            for code, meaning in meanings.items():
                self.assertIn(code + ' = ' + meaning + ';', instructions)
            self.assertEqual(set(request['question']['criteria']), {'KEEP', 'CORRECT', 'CLARIFY'})

    def test_input_only_baseline_has_real_counterexamples_in_every_category(self):
        mod = self.module()
        wrong = Counter()
        for case in mod.build_cases():
            result = mod.baseline(case['task'], case['input'], case['question'])
            self.assertIn(result, case['question']['criteria'])
            wrong[case['task']] += result != case['expected']
        self.assertTrue(all(wrong[task] > 0 for task in {
            'sector_imputation', 'semantic_validation', 'label_qa', 'entity_match',
            'model_routing', 'data_gap_identification'}))
        state = {'evidence': ['Count is minus two; these are events, not money.']}
        q = {'criteria': {'KEEP': '', 'SAFE_NORMALIZE': '', 'FLAG_CONFLICT': '', 'CLARIFY': ''}}
        result = mod.baseline('semantic_validation', state, q)
        tampered = copy.deepcopy(state)
        tampered.update(expected='KEEP', template_group='special', id='known-gold')
        self.assertEqual(mod.baseline('semantic_validation', tampered, q), result)

    def test_support_complete_nulls_and_unknown_units_are_hand_authored(self):
        cases = self.module().build_cases()
        by_premise = {c['template_group']: c for c in cases}
        self.assertEqual(by_premise['semantic_validation:support_structural_null']['expected'], 'SAFE_NORMALIZE')
        self.assertEqual(by_premise['semantic_validation:unknown_scale']['expected'], 'CLARIFY')
        self.assertEqual(by_premise['entity_match:shared_account_sites']['expected'], 'NO_MATCH')
        self.assertEqual(by_premise['data_gap_identification:inventory_not_supplied']['expected'], 'UNKNOWN')
        self.assertEqual(by_premise['model_routing:short_conflict']['expected'], 'STRONG_GPT')

    def test_build_is_reproducible_and_returns_independent_objects(self):
        mod = self.module()
        first = mod.build_cases(seed=11)
        self.assertEqual(first, mod.build_cases(seed=11))
        first[0]['input'].clear()
        self.assertTrue(mod.build_cases(seed=11)[0]['input'])


if __name__ == '__main__':
    unittest.main()
