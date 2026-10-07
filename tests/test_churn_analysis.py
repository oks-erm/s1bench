import copy
import importlib
import json
import unittest


def case(identifier, expected, weight=1, split='evaluation', cluster=None, month='2026-01'):
    return {'id': identifier, 'expected': expected, 'question': {'type': 'noul'},
            'split': split, 'cluster_id': cluster or identifier,
            'metadata': {'sampling_weight': weight,
                         'sampling_stratum': month + ':' + str(int(expected))}}


def row(identifier, value, **changes):
    return dict({'id': identifier, 'model': 'model-a', 'type': 'noul', 'value': value,
                 'valid': True, 'api_error': None, 'phase': 'primary', 'scored': True}, **changes)


class ChurnQualityTests(unittest.TestCase):
    def api(self):
        # Fails explicitly until the new public API exists.
        try:
            return importlib.import_module('churn_analysis')
        except ModuleNotFoundError:
            self.fail('Weighted churn analysis API is not implemented')

    def test_unequal_weights_restore_population_ranking_and_calibration(self):
        # Unit weighting would inflate churn prevalence and change every metric.
        cases = [case('p-high', True, 2), case('p-low', True, 1),
                 case('n-high', False, 4), case('n-low', False, 3)]
        rows = [row('p-high', .9), row('p-low', .2),
                row('n-high', .8), row('n-low', .1)]
        result = self.api().prediction_quality(rows, cases, bootstrap=0)
        self.assertEqual(result['sample']['planned'], 4)
        self.assertEqual(result['population']['represented'], 10)
        self.assertAlmostEqual(result['population']['positive_prevalence'], .3)
        metrics = result['probability_metrics']
        self.assertAlmostEqual(metrics['roc_auc'], 17 / 21)
        self.assertAlmostEqual(metrics['average_precision'], 17 / 21)
        self.assertAlmostEqual(metrics['brier'], .325)
        self.assertAlmostEqual(metrics['log_loss'], .8573992140459266)
        point = result['operating_point']
        self.assertAlmostEqual(point['precision'], 1 / 3)
        self.assertAlmostEqual(point['recall'], 2 / 3)
        self.assertAlmostEqual(point['f1'], 4 / 9)
        self.assertAlmostEqual(point['specificity'], 3 / 7)
        self.assertEqual(point['confusion_weighted'],
                         {'actual_false': {'predicted_false': 3, 'predicted_true': 4, 'invalid': 0},
                          'actual_true': {'predicted_false': 1, 'predicted_true': 2, 'invalid': 0}})

    def test_tied_scores_are_grouped_in_roc_and_step_average_precision(self):
        # Arbitrary input-order tie breaking would claim spurious discrimination.
        cases = [case('p', True, 2), case('n', False, 8)]
        result = self.api().prediction_quality([row('p', .5), row('n', .5)], cases, bootstrap=0)
        self.assertEqual(result['probability_metrics']['roc_auc'], .5)
        self.assertEqual(result['probability_metrics']['average_precision'], .2)
        self.assertEqual(result['operating_point']['recall'], 1)
        self.assertEqual(result['operating_point']['precision'], .2)

    def test_missing_and_invalid_positive_predictions_reduce_planned_recall(self):
        # Dropping failures from recall would turn a 20% capture rate into 100%.
        cases = [case('p-ok', True, 2), case('p-invalid', True, 3),
                 case('p-missing', True, 5), case('n-error', False, 7),
                 case('n-ok', False, 8)]
        rows = [row('p-ok', .8), row('p-invalid', .9, valid=False),
                row('n-error', .1, api_error='TIMEOUT'), row('n-ok', .2),
                row('p-missing', .99, phase='repeat', scored=False)]
        result = self.api().prediction_quality(rows, cases, bootstrap=0)
        self.assertEqual(result['sample'],
                         {'planned': 5, 'positive': 3, 'negative': 2, 'attempted': 4,
                          'valid_probability': 2, 'invalid_attempted': 2, 'unattempted': 1,
                          'customers': 5})
        self.assertEqual(result['coverage']['valid_probability_weight'], 10)
        self.assertEqual(result['coverage']['invalid_attempted_weight'], 10)
        self.assertEqual(result['coverage']['unattempted_weight'], 5)
        self.assertEqual(result['coverage']['weighted_probability_validity'], .4)
        self.assertEqual(result['operating_point']['recall'], .2)
        self.assertEqual(result['operating_point']['specificity'], 8 / 15)
        self.assertEqual(result['operating_point']['precision'], 1)
        self.assertEqual(result['operating_point']['f1'], 1 / 3)
        self.assertEqual(result['operating_point']['confusion_weighted']['actual_true']['invalid'], 8)
        self.assertEqual(result['operating_point']['confusion_sample']['actual_true']['invalid'], 2)
        self.assertAlmostEqual(result['probability_metrics']['brier'], .04)

    def test_probability_endpoints_and_one_class_are_finite_or_unavailable(self):
        # Infinite log loss and fabricated one-class AUC would corrupt JSON reports.
        api = self.api()
        result = api.prediction_quality([row('n', 1)], [case('n', False)], bootstrap=0)
        self.assertIsNone(result['probability_metrics']['roc_auc'])
        self.assertIsNone(result['probability_metrics']['average_precision'])
        self.assertIsNone(result['operating_point']['recall'])
        self.assertEqual(result['probability_metrics']['brier'], 1)
        self.assertAlmostEqual(result['probability_metrics']['log_loss'], 34.53957599234088)
        json.dumps(result, allow_nan=False)
        result = api.prediction_quality([row('p', 0)], [case('p', True)], bootstrap=0)
        self.assertEqual(result['probability_metrics']['average_precision'], 1)
        self.assertIsNone(result['operating_point']['precision'])
        self.assertEqual(result['operating_point']['f1'], 0)
        self.assertIsNone(result['operating_point']['specificity'])

    def test_empty_and_all_missing_cohorts_do_not_fabricate_probability_scores(self):
        # Returning zeros for unavailable discrimination would look like measured failure.
        for cases in [[], [case('p', True, 2), case('n', False, 8)]]:
            result = self.api().prediction_quality([], cases, bootstrap=0)
            self.assertTrue(all(result['probability_metrics'][name] is None
                                for name in ['roc_auc', 'average_precision', 'brier', 'log_loss']))
            json.dumps(result, allow_nan=False)

    def test_invalid_values_are_reviewed_even_when_row_claims_valid(self):
        # A Boolean, NaN, infinity or out-of-range score must never become a decision.
        api = self.api()
        for value in [None, True, False, float('nan'), float('inf'), -.01, 1.01, '0.8']:
            with self.subTest(value=value):
                result = api.prediction_quality([row('p', value)], [case('p', True)], bootstrap=0)
                self.assertEqual(result['coverage']['weighted_probability_validity'], 0)
                self.assertEqual(result['operating_point']['recall'], 0)
                self.assertEqual(result['operating_point']['confusion_weighted']['actual_true']['invalid'], 1)

    def test_rejects_ambiguous_cohorts_and_invalid_sampling_contracts(self):
        # Pooling splits, duplicate primary observations or bad weights invalidates evidence.
        api = self.api()
        base = [case('p', True), case('n', False)]
        malformed = []
        for field, value in [('expected', 1), ('split', 'unknown'), ('cluster_id', '')]:
            altered = copy.deepcopy(base)
            altered[0][field] = value
            malformed.append(altered)
        for value in [0, -1, True, float('nan'), float('inf'), '2', 10 ** 400]:
            altered = copy.deepcopy(base)
            altered[0]['metadata']['sampling_weight'] = value
            malformed.append(altered)
        malformed += [[base[0], base[0]], [base[0], case('n', False, split='development')],
                      [case('p', True, 1e308), case('n', False, 1e308)]]
        for cases in malformed:
            with self.subTest(cases=cases):
                with self.assertRaises(ValueError):
                    api.prediction_quality([], cases, bootstrap=0)
        for rows in [[row('p', .8), row('p', .7)], [row('unknown', .5)],
                     [row('p', .8), row('n', .2, model='model-b')]]:
            with self.assertRaises(ValueError):
                api.prediction_quality(rows, base, bootstrap=0)
        for threshold in [True, -.1, 1.1, float('nan'), '0.5']:
            with self.assertRaises(ValueError):
                api.prediction_quality([], base, threshold=threshold, bootstrap=0)

    def test_customer_spanning_strata_does_not_get_false_independent_intervals(self):
        # Resampling monthly rows separately would understate repeated-customer dependence.
        cases = [case('p', True, cluster='shared'), case('n', False, cluster='shared')]
        result = self.api().prediction_quality([row('p', .8), row('n', .2)], cases)
        self.assertEqual(result['sample']['customers'], 1)
        self.assertIsNone(result['uncertainty']['intervals95'])
        self.assertIn('span', result['uncertainty']['limitation'])

    def test_stratified_customer_bootstrap_preserves_population_and_repeats(self):
        # A row bootstrap would separate same-customer observations in a stratum.
        cases = [case('p1', True, 2, cluster='p'), case('p2', True, 2, cluster='p'),
                 case('p3', True, 2), case('p4', True, 2),
                 case('n1', False, 8, cluster='n'), case('n2', False, 8, cluster='n'),
                 case('n3', False, 8), case('n4', False, 8)]
        rows = [row(c['id'], .8 if c['expected'] else .2) for c in cases]
        api = self.api()
        result = api.prediction_quality(rows, cases, bootstrap=80, seed=9)
        uncertainty = result['uncertainty']
        self.assertEqual(uncertainty['replicates'], 80)
        self.assertEqual(uncertainty['intervals95']['recall'], [1, 1])
        for endpoint in uncertainty['intervals95']['roc_auc']:
            self.assertAlmostEqual(endpoint, 1)
        self.assertAlmostEqual(uncertainty['effective_sample_size'], 5.882352941176471)
        self.assertEqual(result, api.prediction_quality(rows, cases, bootstrap=80, seed=9))

    def test_large_finite_population_weights_do_not_overflow_metrics(self):
        # Multiplying survey weights can overflow despite a finite population total.
        cases = [case('p', True, 5e307), case('n', False, 5e307)]
        result = self.api().prediction_quality([row('p', .8), row('n', .2)], cases, bootstrap=0)
        self.assertEqual(result['probability_metrics']['roc_auc'], 1)
        self.assertEqual(result['operating_point']['f1'], 1)
        json.dumps(result, allow_nan=False)

    def test_missing_sampling_metadata_is_rejected_with_a_contract_error(self):
        # Malformed metadata should not escape validation as an attribute failure.
        cases = [case('p', True)]
        cases[0]['metadata'] = None
        with self.assertRaises(ValueError):
            self.api().prediction_quality([], cases, bootstrap=0)

    def test_bootstrap_moves_all_observations_of_a_customer_together(self):
        # Treating three observations as independent narrows this interval incorrectly.
        cases = [case('bad' + str(index), True, cluster='bad') for index in range(3)]
        cases += [case('good' + str(index), True, cluster='good') for index in range(3)]
        cases += [case('n1', False), case('n2', False)]
        rows = [row(c['id'], .9 if c['cluster_id'] == 'good' else .1) for c in cases]
        result = self.api().prediction_quality(rows, cases, bootstrap=300, seed=42)
        low, high = result['uncertainty']['intervals95']['brier']
        self.assertAlmostEqual(low, .01)
        self.assertAlmostEqual(high, .61)


class ChurnThresholdTests(unittest.TestCase):
    def api(self):
        return ChurnQualityTests().api()

    def test_weighted_development_f1_selects_exact_threshold_and_accounts_for_missing(self):
        # An unweighted threshold would trade costly false positives for a rare extra churner.
        cases = [case('p-high', True, 1, 'development'), case('p-low', True, 1, 'development'),
                 case('n-high', False, 10, 'development'), case('p-missing', True, 2, 'development')]
        result = self.api().choose_threshold([row('p-high', .9), row('p-low', .4), row('n-high', .7)], cases)
        self.assertEqual(result['threshold'], .9)
        self.assertEqual(result['status'], 'selected')
        self.assertEqual(result['development_f1'], .4)
        self.assertEqual(result['selection_basis'], 'development_weighted_f1_experimental')

    def test_threshold_ties_prefer_higher_and_evaluation_cannot_tune(self):
        # Tied F1 should avoid extra negatives and evaluation must remain unseen by selection.
        api = self.api()
        cases = [case('p', True, split='development'), case('n', False, split='development')]
        result = api.choose_threshold([row('p', .8), row('n', .2)], cases)
        self.assertEqual(result['threshold'], .8)
        with self.assertRaises(ValueError):
            api.choose_threshold([row('p', .8)], [case('p', True)])
        result = api.choose_threshold([], cases)
        self.assertIsNone(result['threshold'])
        self.assertEqual(result['status'], 'no_valid_probabilities')


if __name__ == '__main__':
    unittest.main()
