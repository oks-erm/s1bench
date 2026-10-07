"""Population-weighted predictive analysis for sampled monthly churn cases.

No model calls, input feature selection, business policy, or evaluation tuning happen
here. Sampling metadata is used only for analysis. Probability metrics describe the
valid-response subset; coverage and operating-point recall retain all planned cases.
"""

from collections import defaultdict
import math
import random
import re

import benchmark as b


LOG_LOSS_EPSILON = 1e-15
METRIC_SOURCES = {
    'average_precision': 'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html',
    'roc_auc': 'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html',
}


def _finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _observations(records, cases):
    """Validate one alias/cohort, then join predictions onto the planned cases."""
    by_id, splits = {}, set()
    for case in cases:
        identifier = case.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise ValueError('Case IDs must be unique nonempty strings.')
        if type(case.get('expected')) is not bool:
            raise ValueError('Churn expected labels must be Boolean.')
        if case.get('question', {}).get('type') != 'noul':
            raise ValueError('Churn cases must have noul questions.')
        split = case.get('split')
        if split not in {'development', 'evaluation'}:
            raise ValueError('Churn cases require development or evaluation split.')
        splits.add(split)
        cluster = case.get('cluster_id')
        if not isinstance(cluster, str) or not cluster:
            raise ValueError('Cases require a nonempty customer cluster_id.')
        metadata = case.get('metadata', {})
        if not isinstance(metadata, dict):
            raise ValueError('Cases require sampling metadata objects.')
        weight = metadata.get('sampling_weight')
        if not _finite_number(weight) or weight <= 0:
            raise ValueError('Sampling weights must be positive finite numbers.')
        stratum = metadata.get('sampling_stratum')
        if not isinstance(stratum, str) or not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2]):[01]', stratum):
            raise ValueError('Sampling strata must be YYYY-MM:0 or YYYY-MM:1.')
        if stratum[-1] != str(int(case['expected'])):
            raise ValueError('Sampling stratum label must match the expected label.')
        by_id[identifier] = {'id': identifier, 'positive': case['expected'],
                             'weight': float(weight), 'stratum': stratum, 'cluster': cluster}
    if len(splits) > 1:
        raise ValueError('Do not pool development and evaluation churn cases.')

    predictions, models = {}, set()
    for row in records:
        if row.get('id') not in by_id:
            raise ValueError('Prediction references an unexpected case ID.')
        if not b.primary_row(row):
            continue
        identifier = row['id']
        if identifier in predictions:
            raise ValueError('Duplicate primary predictions for a case.')
        model = row.get('model')
        if not isinstance(model, str) or not model:
            raise ValueError('Primary predictions require a model alias.')
        models.add(model)
        predictions[identifier] = row
    if len(models) > 1:
        raise ValueError('Analyze one model alias at a time.')

    observations = []
    for identifier, planned in by_id.items():
        row = predictions.get(identifier)
        probability = row.get('value') if row is not None else None
        valid = (row is not None and row.get('type') == 'noul' and row.get('valid') is True
                 and not row.get('api_error') and _finite_number(probability)
                 and 0 <= probability <= 1)
        observations.append(dict(planned, attempted=row is not None,
                                 probability=float(probability) if valid else None))
    try:
        total_weight = math.fsum(item['weight'] for item in observations)
    except OverflowError as exc:
        raise ValueError('Total represented population must be finite.') from exc
    if not math.isfinite(total_weight):
        raise ValueError('Total represented population must be finite.')
    return observations, next(iter(splits), None), next(iter(models), None)


def _probability_metrics(observations):
    available = [item for item in observations if item['probability'] is not None]
    total = math.fsum(item['weight'] for item in available)
    metrics = {'roc_auc': None, 'average_precision': None, 'brier': None, 'log_loss': None}
    if not total:
        return metrics
    # Normalized weights preserve these ratios and avoid overflowing pair products
    # or weighted loss sums when represented population weights are very large.
    available = [dict(item, weight=item['weight'] / total) for item in available]
    total = math.fsum(item['weight'] for item in available)
    positive = math.fsum(item['weight'] for item in available if item['positive'])
    negative = math.fsum(item['weight'] for item in available if not item['positive'])
    metrics['brier'] = math.fsum(item['weight'] * (item['probability'] - item['positive']) ** 2
                                  for item in available) / total
    losses = []
    for item in available:
        probability = min(1 - LOG_LOSS_EPSILON, max(LOG_LOSS_EPSILON, item['probability']))
        loss = -math.log(probability) if item['positive'] else -math.log1p(-probability)
        losses.append(item['weight'] * loss)
    metrics['log_loss'] = math.fsum(losses) / total

    # Equal probabilities enter together: ROC awards half credit to tied pairs,
    # and AP uses precision after the full tied group enters (no interpolation).
    groups = defaultdict(lambda: [0.0, 0.0])
    for item in available:
        groups[item['probability']][0 if item['positive'] else 1] += item['weight']
    negative_below = pair_credit = 0.0
    for probability in sorted(groups):
        group_positive, group_negative = groups[probability]
        pair_credit += group_positive * (negative_below + .5 * group_negative)
        negative_below += group_negative
    if positive and negative:
        metrics['roc_auc'] = pair_credit / (positive * negative)
    cumulative_positive = cumulative_total = ap = 0.0
    if positive:
        for probability in sorted(groups, reverse=True):
            group_positive, group_negative = groups[probability]
            cumulative_positive += group_positive
            cumulative_total += group_positive + group_negative
            ap += (group_positive / positive) * (cumulative_positive / cumulative_total)
        metrics['average_precision'] = ap
    return metrics


def _operating_point(observations, threshold):
    weighted = {label: {'predicted_false': 0.0, 'predicted_true': 0.0, 'invalid': 0.0}
                for label in ('actual_false', 'actual_true')}
    counts = {label: {'predicted_false': 0, 'predicted_true': 0, 'invalid': 0}
              for label in ('actual_false', 'actual_true')}
    for item in observations:
        actual = 'actual_true' if item['positive'] else 'actual_false'
        probability = item['probability']
        predicted = ('invalid' if probability is None else
                     'predicted_true' if probability >= threshold else 'predicted_false')
        weighted[actual][predicted] += item['weight']
        counts[actual][predicted] += 1
    true_positive = weighted['actual_true']['predicted_true']
    false_positive = weighted['actual_false']['predicted_true']
    planned_positive = math.fsum(weighted['actual_true'].values())
    planned_negative = math.fsum(weighted['actual_false'].values())
    return {'threshold': threshold, 'comparison': 'probability >= threshold',
            'precision': _ratio(true_positive, true_positive + false_positive),
            'recall': _ratio(true_positive, planned_positive),
            'f1': _ratio(true_positive, .5 * planned_positive + .5 * true_positive + .5 * false_positive),
            'specificity': _ratio(weighted['actual_false']['predicted_false'], planned_negative),
            'confusion_weighted': weighted, 'confusion_sample': counts,
            'recall_denominator': 'all planned weighted positives, including invalid and unattempted',
            'specificity_denominator': 'all planned weighted negatives, including invalid and unattempted',
            'precision_denominator': 'weighted valid predicted positives only'}


def _percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _uncertainty(observations, threshold, bootstrap, seed):
    weights = [item['weight'] for item in observations]
    total = math.fsum(weights)
    # Normalize before squaring so finite large survey weights do not overflow.
    effective = 1 / math.fsum((weight / total) ** 2 for weight in weights) if total else None
    result = {'method': 'stratified_customer_cluster_percentile_bootstrap',
              'requested_replicates': bootstrap, 'replicates': 0, 'seed': seed,
              'effective_sample_size': effective,
              'effective_sample_size_definition': 'Kish weighted row size, not independent customer count',
              'intervals95': None, 'available_replicates': {},
              'conditional_on': 'observed customer clusters, fixed sampling strata and represented stratum totals; fixed threshold',
              'limitation': None}
    if not bootstrap or not observations:
        result['limitation'] = 'Bootstrap disabled or no planned cases.'
        return result
    cluster_strata = defaultdict(set)
    strata = defaultdict(lambda: defaultdict(list))
    for item in observations:
        cluster_strata[item['cluster']].add(item['stratum'])
        strata[item['stratum']][item['cluster']].append(item)
    if any(len(group) > 1 for group in cluster_strata.values()):
        result['limitation'] = ('Customer clusters span sampling strata; intervals omitted because independently '
                                'resampling strata would break customer dependence. A survey-design-aware '
                                'bootstrap is required.')
        return result
    if any(len(clusters) < 2 for clusters in strata.values()):
        result['limitation'] = 'Intervals omitted: at least one sampling stratum has fewer than two customer clusters.'
        return result
    rng = random.Random(seed)
    samples = defaultdict(list)
    prepared = [(list(clusters.values()), math.fsum(item['weight'] for group in clusters.values() for item in group))
                for _, clusters in sorted(strata.items())]
    for _ in range(bootstrap):
        replicate = []
        for clusters, population in prepared:
            drawn = [item for _ in range(len(clusters)) for item in rng.choice(clusters)]
            largest = max(item['weight'] for item in drawn)
            drawn_weight = math.fsum(item['weight'] / largest for item in drawn)
            replicate.extend(dict(item, weight=population * ((item['weight'] / largest) / drawn_weight))
                             for item in drawn)
        probability = _probability_metrics(replicate)
        point = _operating_point(replicate, threshold)
        for name, value in dict(probability, **{key: point[key] for key in ('precision', 'recall', 'f1', 'specificity')}).items():
            if value is not None:
                samples[name].append(value)
    names = ('roc_auc', 'average_precision', 'brier', 'log_loss', 'precision', 'recall', 'f1', 'specificity')
    result['replicates'] = bootstrap
    result['available_replicates'] = {name: len(samples[name]) for name in names}
    # Do not silently condition an interval on the subset of replicates where its
    # denominator happens to exist. Missing metrics invalidate that interval.
    result['intervals95'] = {name: [_percentile(samples[name], .025), _percentile(samples[name], .975)]
                             if len(samples[name]) == bootstrap else None for name in names}
    result['limitation'] = ('Approximate conditional intervals; no finite-population correction, sampling-design '
                            'uncertainty, threshold-selection uncertainty, or future-month distribution shift. '
                            'Few independent customers can make percentile intervals unreliable.')
    return result


def prediction_quality(records, cases, threshold=.5, bootstrap=300, seed=42):
    """Analyze one model and one split, restoring population weights after sampling.

    ROC-AUC, step-area average precision, Brier and clipped binary log loss use only
    finite valid P(churn) values. Always interpret them alongside weighted coverage.
    Missing/invalid predictions are abstentions; they reduce planned positive recall
    and negative specificity. An undefined denominator yields JSON null, not NaN.
    """
    if not _finite_number(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be a finite number in [0, 1].')
    if type(bootstrap) is not int or not 0 <= bootstrap <= 300:
        raise ValueError('Bootstrap replicates must be an integer from 0 to 300.')
    if type(seed) is not int:
        raise ValueError('Bootstrap seed must be an integer.')
    observations, split, model = _observations(records, cases)
    total = math.fsum(item['weight'] for item in observations)
    positive = math.fsum(item['weight'] for item in observations if item['positive'])
    valid = [item for item in observations if item['probability'] is not None]
    invalid = [item for item in observations if item['attempted'] and item['probability'] is None]
    missing = [item for item in observations if not item['attempted']]
    valid_weight = math.fsum(item['weight'] for item in valid)
    return {'split': split, 'model': model,
            'sample': {'planned': len(observations), 'positive': sum(item['positive'] for item in observations),
                       'negative': sum(not item['positive'] for item in observations),
                       'attempted': len(observations) - len(missing), 'valid_probability': len(valid),
                       'invalid_attempted': len(invalid), 'unattempted': len(missing),
                       'customers': len({item['cluster'] for item in observations})},
            'population': {'represented': total, 'positive': positive, 'negative': total - positive,
                           'positive_prevalence': _ratio(positive, total)},
            'coverage': {'valid_probability_weight': valid_weight,
                         'invalid_attempted_weight': math.fsum(item['weight'] for item in invalid),
                         'unattempted_weight': math.fsum(item['weight'] for item in missing),
                         'weighted_probability_validity': _ratio(valid_weight, total),
                         'sample_probability_validity': _ratio(len(valid), len(observations)),
                         'denominator': 'all planned cases'},
            'probability_metrics': dict(_probability_metrics(observations),
                                        denominator='valid probability subset only',
                                        average_precision_convention='noninterpolated step area; tied scores enter together',
                                        roc_auc_convention='weighted positive-negative pairs; half credit for equal scores',
                                        log_loss_epsilon=LOG_LOSS_EPSILON,
                                        sources=dict(METRIC_SOURCES)),
            'operating_point': _operating_point(observations, threshold),
            'uncertainty': _uncertainty(observations, threshold, bootstrap, seed)}


def choose_threshold(devrecords, devcases):
    """Select an experimental F1 point on development data, never a business policy."""
    cases = list(devcases)
    if any(case.get('split') != 'development' for case in cases):
        raise ValueError('Threshold selection is restricted to development cases.')
    observations, _, model = _observations(devrecords, cases)
    probabilities = {item['probability'] for item in observations if item['probability'] is not None}
    base = {'threshold': None, 'status': 'no_valid_probabilities',
            'selection_basis': 'development_weighted_f1_experimental', 'model': model,
            'business_approval': False, 'development_f1': None, 'candidate_count': 0,
            'tie_break': 'higher threshold', 'invalid_predictions': 'remain review/abstention'}
    if not probabilities:
        return base
    if not any(item['positive'] for item in observations):
        return dict(base, status='no_development_positives')
    candidates = sorted(probabilities | {0.0, 1.0})
    score, selected = max((_operating_point(observations, threshold)['f1'], threshold)
                          for threshold in candidates)
    return dict(base, threshold=selected, status='selected', development_f1=score,
                candidate_count=len(candidates))
