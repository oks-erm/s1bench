"""Saved-response decision analysis. No inference, network calls or mutations of evidence."""
from collections import Counter, defaultdict
import copy
import math
import statistics

REVIEW_LABELS = ['NONE', 'CLARIFY', 'ESCALATE', 'ABSTAIN', 'HUMAN', 'HUMAN_REVIEW']
RISK_VERSION = 'directional-choice-v1'


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def probability(x):
    return finite(x) and 0 <= x <= 1


def validate_policy(policy):
    if policy.get('mode', 'label') not in {'label', 'probability', 'confidence'}:
        raise ValueError('Acceptance mode must be label, probability or confidence.')
    for key in ('threshold', 'negative_threshold', 'positive_threshold'):
        if key in policy and not probability(policy[key]):
            raise ValueError(key + ' must be a finite number from 0 to 1.')
    if policy.get('negative_threshold', .1) >= policy.get('positive_threshold', .9):
        raise ValueError('Negative threshold must be below positive threshold.')


def acceptance(row, policy=None, business=False):
    policy = policy or {}
    validate_policy(policy)
    if row.get('api_error') or not row.get('valid'):
        return 'failed'
    if row.get('type') == 'score':
        return 'reviewed'
    if row.get('type') == 'choice' and row.get('decision') in policy.get('review_labels', REVIEW_LABELS):
        return 'reviewed'
    if business and (policy.get('business_approved') is not True or
                     row.get('decision') in policy.get('ineligible_labels', [])):
        return 'reviewed'
    mode = policy.get('mode', 'label')
    if mode == 'label':
        return 'accepted' if row.get('type') == 'choice' else 'reviewed'
    if policy.get('accept_none'):
        return 'reviewed'
    if row.get('type') == 'noul':
        value = row.get('value')
        return 'accepted' if probability(value) and (value <= policy.get('negative_threshold', .1) or
                                                      value >= policy.get('positive_threshold', .9)) else 'reviewed'
    value = row.get('confidence')
    if mode == 'probability':
        probs = row.get('probabilities')
        valid_vector = (isinstance(probs, dict) and probs and
                        all(probability(v) for v in probs.values()) and abs(sum(probs.values())-1) <= 1e-5)
        value = probs.get(row.get('decision')) if valid_vector else None
    return 'accepted' if probability(value) and value >= policy.get('threshold', .9) else 'reviewed'


def directional(row):
    return bool(row.get('valid') and not row.get('api_error') and
                row.get('decision') in row.get('critical_error_choices', []))


def acceptance_summary(rows, policy=None, planned=None, business=False):
    import benchmark as b
    n = len(rows) if planned is None else planned
    groups = defaultdict(list)
    for row in rows:
        groups[acceptance(row, policy, business)].append(row)
    accepted = groups['accepted']
    errors = sum(not r['correct'] for r in accepted)
    lo, hi, method = b.success_interval(accepted, samples=300) if accepted else (None, None, 'no accepted cases')
    # A degenerate family bootstrap cannot establish zero future risk.
    families = len({r.get('cluster_id', r['id']) for r in accepted})
    if accepted and (errors == 0 or errors == len(accepted)) and families < len(accepted):
        lo, hi, method = None, None, 'boundary outcome in clustered sample; risk bound unavailable'
    directional_policy = (policy or {}).get('risk_policy') == RISK_VERSION or any('critical_error_choices' in r for r in rows)
    def error(row):
        return directional(row) if directional_policy else bool(row.get('critical') and not row['correct'])
    def exposure(row):
        return bool(row.get('critical_error_choices')) if directional_policy else bool(row.get('critical'))
    exposed = [r for r in accepted if exposure(r)]
    return {'planned': n, 'accepted': len(accepted), 'reviewed': len(groups['reviewed']),
            'failed': len(groups['failed']), 'unattempted': max(0, n-len(rows)),
            'coverage': len(accepted)/n if n else None,
            'review_rate': (n-len(accepted))/n if n else None,
            'accepted_errors': errors, 'accepted_accuracy': 1-errors/len(accepted) if accepted else None,
            'accepted_error_rate': errors/len(accepted) if accepted else None,
            'accepted_families': families, 'accepted_ci_low': lo, 'accepted_ci_high': hi, 'ci_method': method,
            'risk_policy': RISK_VERSION if directional_policy else 'legacy-critical-case-errors',
            'accepted_critical_errors': sum(error(r) for r in accepted),
            'accepted_risk_exposures': len(exposed),
            'accepted_critical_rate': sum(error(r) for r in accepted)/len(exposed) if exposed else None,
            'critical_errors_all_attempts': sum(error(r) for r in rows),
            'risk_exposures_all_attempts': sum(exposure(r) for r in rows)}


def risk_curve(rows, policy=None, planned=None):
    import benchmark as b
    policy = policy or {}
    complete = bool(rows) and len(rows) == (len(rows) if planned is None else planned)
    priced = complete and all(finite(r.get('_api_cost')) for r in rows)
    timed = complete and all(finite(r.get('latency_s')) for r in rows)
    result = []
    for threshold in (0, .5, .6, .7, .8, .9, .95, .99, 1):
        p = threshold_policy(policy,threshold)
        result.append({'threshold': threshold, 'interpretation': 'exploratory on displayed cohort',
                       'policy':p,
                       'model_api_per_1k': sum(r['_api_cost'] for r in rows)/len(rows)*1000 if priced else None,
                       'model_p95_ms': b.percentile([r['latency_s'] for r in rows],.95)*1000 if timed else None,
                       'cost_latency_basis':'Measured model calls only; human/downstream processing excluded',
                       **acceptance_summary(rows, p, planned)})
    return result


def threshold_policy(policy,threshold):
    p = {**policy, 'mode': policy.get('mode', 'probability'), 'accept_none': False}
    if p['mode'] == 'label':p['mode'] = 'probability'
    p.update(threshold=threshold, negative_threshold=min(.49,1-threshold),positive_threshold=max(.51,threshold))
    return p


def cascade_curves(curves,records,summaries,cohort,baseline):
    import benchmark as b
    output=[]
    for curve in curves:
        replay=b.fallback_replay(records,summaries,cohort['dataset'],cohort['task'],baseline,set(),curve['policy'])
        cascade=next((r for r in replay if r['model']==cohort['model']),{})
        output.append({**curve,'cascade_reference':baseline,
                       'simulated_cascade_api_per_1k':cascade.get('api_per_1k'),
                       'simulated_cascade_p95_ms':cascade.get('simulated_p95_ms'),
                       'simulated_cascade_success':cascade.get('success_rate'),
                       'simulated_cascade_fallback_rate':cascade.get('fallback_rate'),
                       'cascade_basis':'Offline serial replay from complete matched primary responses; not measured deployment'})
    return output


def probability_vector(row, labels):
    p = row.get('probabilities')
    if (not row.get('valid') or row.get('api_error') or not isinstance(p, dict) or set(p) != set(labels)
            or any(not probability(v) for v in p.values()) or abs(sum(p.values())-1) > 1e-5):
        return None
    return p


def probability_quality(rows, cases):
    by_id = {c['id']: c for c in cases}
    losses, bins = [], defaultdict(list)
    for row in rows:
        c = by_id.get(row['id'])
        if not c or row['type'] != 'choice':
            continue
        p = probability_vector(row, c['question']['criteria'])
        if p is None:
            continue
        losses.append(sum((v-int(k == row['expected']))**2 for k, v in p.items()))
        score = p[row['decision']]
        bins[min(9, int(score*10))].append((score, int(row['correct'])))
    return {'multiclass_brier': statistics.mean(losses) if losses else None,
            'probability_cases': len(losses), 'probability_coverage': len(losses)/len(rows) if rows else None,
            'reliability': [{'bin_lower': k/10, 'cases': len(v), 'mean_probability': statistics.mean(x[0] for x in v),
                             'accuracy': statistics.mean(x[1] for x in v), 'warning': 'descriptive; not calibration certification'}
                            for k, v in sorted(bins.items())]}


def scenario(a, settings, *, api_per_1k=None, hosting_per_1k=None, api_currency='USD'):
    """Minutes and currency/unit assumptions; missing values propagate as unavailable."""
    s = settings
    for key in ('volume', 'manual_minutes', 'review_minutes', 'audit_fraction', 'audit_minutes',
                'rework_minutes', 'labour_per_hour', 'setup_cost', 'recurring_cost', 'fx_api_to_labour'):
        value = s.get(key)
        if value is not None and (not finite(value) or value < 0):
            raise ValueError(key + ' must be nonnegative and finite.')
    if s.get('audit_fraction', 0) is not None and s.get('audit_fraction', 0) > 1:
        raise ValueError('Audit fraction must be between 0 and 1.')
    if s.get('fx_api_to_labour') is not None and s['fx_api_to_labour'] <= 0:
        raise ValueError('Currency conversion must be positive when supplied.')
    n, volume = a.get('planned', 0), s.get('volume')
    usable = n > 0 and finite(volume) and a.get('unattempted', 0) == 0
    def hours(count, minutes):
        return count/n*volume*minutes/60 if usable and finite(minutes) and finite(count) else None
    manual = hours(n, s.get('manual_minutes'))
    review = hours(n-a.get('accepted', 0), s.get('review_minutes'))
    audit = hours(a.get('accepted', 0)*s['audit_fraction'], s.get('audit_minutes')) if finite(s.get('audit_fraction')) else None
    rework = hours(a.get('accepted_errors', 0), s.get('rework_minutes', 0))
    effort = sum((review, audit, rework)) if all(finite(x) for x in (review, audit, rework)) else None
    gross = manual-review if finite(manual) and finite(review) else None
    net = manual-effort if finite(manual) and finite(effort) else None
    fx = 1 if api_currency == s.get('currency') else s.get('fx_api_to_labour')
    labour = effort*s['labour_per_hour'] if finite(effort) and finite(s.get('labour_per_hour')) else None
    tech = (api_per_1k+hosting_per_1k)*volume/1000*fx if usable and all(finite(x) for x in (api_per_1k, hosting_per_1k, fx)) else None
    total = labour+tech+s['recurring_cost'] if all(finite(x) for x in (labour, tech, s.get('recurring_cost'))) else None
    baseline = manual*s['labour_per_hour'] if finite(manual) and finite(s.get('labour_per_hour')) else None
    savings = baseline-total if finite(baseline) and finite(total) else None
    return {'manual_hours': manual, 'review_hours': review, 'audit_hours': audit, 'estimated_rework_hours': rework,
            'gross_hours_saved': gross, 'net_hours_saved': net, 'manual_cost': baseline,
            'monthly_total': total, 'total_per_1k': total/volume*1000 if finite(total) and volume > 0 else None,
            'monthly_savings': savings, 'payback_months': s['setup_cost']/savings if finite(s.get('setup_cost')) and finite(savings) and savings > 0 else None,
            'currency': s.get('currency'), 'basis': 'user assumptions; cohort error/prevalence extrapolation; capacity is not guaranteed cash savings'}


def policy_for(cfg, task, alias=None):
    task_cfg = cfg.get('tasks', {}).get(task, {})
    p = {**cfg.get('acceptance', {}), **task_cfg.get('acceptance', {})}
    if 'review_labels' in task_cfg: p['review_labels'] = task_cfg['review_labels']
    p.update(cfg.get('frozen_acceptance', {}).get(alias, {}).get(task, {}))
    if cfg.get('shve_protocol'):p.setdefault('risk_policy',RISK_VERSION)
    return p


def diagnostics(cases, records):
    import benchmark as b
    families, duplicates, templates = defaultdict(list), defaultdict(list), defaultdict(set)
    for c in cases:
        families[c['cluster_id']].append(c)
        duplicates[b.fingerprint([c['task'], b.state_for(c), c['question']])].append(c['id'])
        templates[c['task']].add(c.get('scenario', b.fingerprint(c['question'])))
    cross = []
    for family, members in families.items():
        roles = sorted({c.get('split', c.get('evaluation_role', c.get('dataset_role','unspecified'))) for c in members})
        boundary_roles = {'development' if r=='dev' else 'evaluation' if r in {'holdout','test'} else r
                          for r in roles if r in {'dev','development','calibration','holdout','test','evaluation'}}
        datasets = sorted({c['dataset'] for c in members})
        if len(datasets) > 1 or len(roles) > 1:
            cross.append({'family': family, 'datasets': datasets, 'roles': roles,
                          'boundary_overlap': len(boundary_roles) > 1})
    groups = defaultdict(list)
    for r in records:
        if b.primary_row(r): groups[(r['model'], r['dataset'], r['task'], r['cluster_id'])].append(r)
    errors = [{'model': m, 'dataset': ds, 'task': t, 'family': f, 'cases': len(rr),
               'errors': sum(not r['correct'] for r in rr)} for (m, ds, t, f), rr in groups.items() if any(not r['correct'] for r in rr)]
    subgroups = defaultdict(list)
    index = {c['id']: c for c in cases}
    for r in records:
        if not b.primary_row(r): continue
        c = index[r['id']]
        for key in ('language', 'BU', 'source'):
            metadata = c.get('metadata') if isinstance(c.get('metadata'), dict) else {}
            state = b.state_for(c)
            if isinstance(state, str):
                try: state = b.parse(state)
                except (ValueError, TypeError): state = {}
            state = state if isinstance(state, dict) else {}
            value = c.get(key, metadata.get(key, state.get(key)))
            if value is not None and isinstance(value, (str, int)):
                subgroups[(r['model'], r['dataset'], r['task'], key, str(value))].append(r)
    return {'primary_cases': len(cases), 'primary_requests':sum(b.primary_row(r) for r in records),
            'primary_requests_by_model':dict(Counter(r['model'] for r in records if b.primary_row(r))),
            'repeat_requests': sum(r.get('phase') == 'repeat' for r in records),
            'unique_declared_families': len(families), 'family_overlap': cross,
            'duplicate_input_groups': [v for v in duplicates.values() if len(v) > 1],
            'declared_templates': {k: len(v) for k, v in templates.items()},
            'error_concentration': sorted(errors, key=lambda r: -r['errors']),
            'subgroups': [{'model': m, 'dataset': ds, 'task': t, 'field': k, 'value': v, 'cases': len(rr),
                           'families': len({r['cluster_id'] for r in rr}), 'success_rate': sum(r['correct'] for r in rr)/len(rr),
                           'warning': 'Small/descriptive subgroup; source independence unverified'}
                          for (m, ds, t, k, v), rr in subgroups.items()]}


def baseline_comparisons(summaries, records, rule_summaries, rule_records, *, samples=300, seed=42):
    """Paired model-minus-rule differences only for complete matching cohorts."""
    import benchmark as b
    needs = {'sector_imputation':'Complete missing sector labels from activity descriptions',
             'label_qa':'Review unsupported or incorrect existing sector labels',
             'semantic_validation':'Check numeric consistency and explicit unit assumptions',
             'entity_match':'Propose delivery-point matches for human review',
             'vision_routing':'Classify requests under the supplied routing policy',
             'data_gap_identification':'Identify missing, stale or insufficient source history'}
    rules = {(r['dataset'],r['task']):r for r in rule_summaries}
    output = []
    for summary in summaries:
        key = (summary['dataset'],summary['task'])
        rule = rules.get(key)
        if rule is None:
            continue
        reference = next((r for r in summaries if (r['dataset'],r['task'])==key and r['model']=='gpt' and r['status']=='complete'),None)
        strongest = ('GPT' if reference['success_rate'] > rule['success_rate'] else
                     'Deterministic rules and GPT tied' if reference['success_rate'] == rule['success_rate'] else
                     'Deterministic rules') if reference else 'Deterministic rules; GPT comparison pending'
        difference, low, high, method = None, None, None, 'incomplete model cohort'
        if summary['status'] == rule['status'] == 'complete':
            left = [r for r in records if b.primary_row(r) and
                    (r['model'],r['dataset'],r['task']) == (summary['model'],*key)]
            right = [r for r in rule_records if (r['dataset'],r['task']) == key]
            low, high, method = b.paired_interval(left,right,samples,seed)
            if set(r['id'] for r in left) == set(r['id'] for r in right):
                difference = summary['success_rate']-rule['success_rate']
        output.append({'dataset':key[0],'task':key[1],'model':summary['model'],
                       'business_need':needs.get(key[1],key[1]),'status':summary['status'],
                       'strongest_baseline':strongest,
                       'success_rate':summary['success_rate'],'rules_success_rate':rule['success_rate'],
                       'incremental_accuracy':difference,'delta_ci_low':low,'delta_ci_high':high,
                       'comparison_method':method,'families':summary['families'],
                       'review_demand':summary['acceptance']['review_rate'],
                       'directional_critical_errors':summary['critical_failures'],
                       'risk_exposures':summary.get('risk_exposures'),
                       'deployment_constraints':summary['deployment_recommendation'],
                       'next_validation':('Fresh topic-and-complexity cases with declared target capabilities; target execution is needed only for a separate downstream-quality study'
                                          if key[1]=='vision_routing' else
                                          'Human-adjudicated unseen templates, verified source independence and operational review/audit costs')})
    return output


def enrich_summary(summary, rows, cases, cfg, profile):
    p = policy_for(cfg, summary['task'], summary['model'])
    semantic_policy = policy_for(cfg, summary['task'])
    stats = acceptance_summary(rows, p, summary['planned'])
    eligible = acceptance_summary(rows, p, summary['planned'], business=True)
    label = acceptance_summary(rows, {**semantic_policy, 'mode': 'label'}, summary['planned'])
    biz = {**cfg.get('business', {}), **cfg.get('business', {}).get('use_cases', {}).get(summary['task'], {})}
    status = profile.get('deployment_eligibility', 'unknown')
    outstanding = [k for k in ('max_risk_percent', 'max_p95_ms', 'min_consistency_percent') if biz.get(k) is None]
    blocked_decisions = {'max_risk_percent':'No acceptable harmful-error target to assess pilot risk',
                         'max_p95_ms':'No latency target to assess workflow responsiveness',
                         'min_consistency_percent':'No repeat-agreement target to assess repeatability'}
    by_label = []
    labels = sorted({str(label) for c in cases for label in c['question'].get('criteria', {})} |
                    {str(r['expected']) for r in rows})
    for label_name in labels:
        gold = [r for r in rows if str(r['expected']) == label_name]
        predicted = [r for r in rows if r.get('valid') and not r.get('api_error') and str(r['decision']) == label_name]
        tp = sum(r['correct'] for r in gold)
        by_label.append({'label': label_name, 'support': len(gold), 'predicted': len(predicted),
                         'precision': tp/len(predicted) if predicted else None, 'recall': tp/len(gold) if gold else None})
    review_gold = [r for r in rows if r['expected'] in semantic_policy.get('review_labels', REVIEW_LABELS)]
    # Every confusion direction remains inspectable, including directions outside proposed lists.
    directions = Counter((str(r['expected']), str(r['decision']) if r.get('valid') and not r.get('api_error') else '(failure)') for r in rows if not r['correct'])
    ambiguous = {c['id'] for c in cases if c.get('adjudication_required') or
                 (c.get('task') == 'retrieval' and c.get('cluster_id') == 'synthetic:retrieval:pattern-7')}
    retained = [r for r in rows if r['id'] not in ambiguous]
    summary['adjudication'] = {'flagged_cases': len(ambiguous), 'remaining_attempts': len(retained),
        'success_without_flagged': sum(r['correct'] for r in retained)/len(retained) if retained else None,
        'basis': 'Sensitivity analysis only; historical gold and primary scores unchanged. Flagged ambiguity requires human adjudication.'}
    summary.update(acceptance=stats, business_acceptance=eligible, acceptance_policy=p,
                   label_based_coverage=label['coverage'], threshold_coverage=stats['coverage'] if p.get('mode', 'label') != 'label' else None,
                   business_eligible_coverage=eligible['coverage'], per_label=by_label,
                   review_recall=sum(r.get('valid') and not r.get('api_error') and r.get('decision') in semantic_policy.get('review_labels', REVIEW_LABELS) for r in review_gold)/len(review_gold) if review_gold else None,
                   review_exposures=len(review_gold),
                   error_directions=[{'gold': g, 'prediction': v, 'errors': n, 'gold_exposures': sum(str(r['expected']) == g for r in rows)} for (g, v), n in sorted(directions.items())],
                   evaluation_scope=cases[0].get('split', cases[0].get('evaluation_role', 'unspecified')),
                   deployment_eligibility=status, outstanding_targets=outstanding,
                   blocked_business_decisions=[blocked_decisions[k] for k in outstanding],
                   use_case_overrides=cfg.get('business', {}).get('use_cases', {}).get(summary['task'], {}),
                   probability_quality=probability_quality(rows, cases))
    summary['workflow_scenario'] = scenario(stats, {**cfg.get('workflow_cost', {}), 'volume': cfg.get('workflow_cost', {}).get('volume', biz.get('monthly_volume'))},
                                           api_per_1k=summary['api_per_1k'] if summary['cost_coverage'] == 1 else None,
                                           hosting_per_1k=summary['active_hosting_per_1k'], api_currency=summary['currency'])
    summary['deployment_recommendation'] = 'Not approved for deployment' if status == 'not approved' else 'Organisational approval required' if status != 'approved' else 'Approval declared; quality and workflow validation still required'
    return summary
