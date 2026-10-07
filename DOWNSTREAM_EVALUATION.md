# Creating downstream evidence

The SHVE scenario run measures decisions under supplied rubrics. Its source sample has
600 rows and no observed churn outcome. Routing examples name classes of execution
routes, not executable specialist deployments. We need separate data and operational
measurements to establish downstream effects. Keep this experiment separate from the
current frozen run; do not replace its labels or claim a fresh holdout after tuning.

## Routing experiment

### Topic and complexity target selection

Target selection is a standalone classification benchmark. Define target roles and
their capabilities in the rubric: a typed model for bounded decisions, a weaker
GPT for straightforward grounded transformations, and a stronger GPT for dependent
reasoning, conflicting evidence or multi-constraint synthesis. These are declared
experimental capabilities, not measured performance guarantees. Exact deployment
IDs and calls to the target models are unnecessary for this test.

Use synthetic energy requests with different complexity levels within each topic,
including short difficult requests and long straightforward ones. Freeze the
selection policy and gold routes before inference. Measure target-selection
accuracy, per-target precision/recall, confusion matrices, stronger-target misses,
unnecessary escalation and paired paraphrase consistency. Calling the models being
evaluated as selectors has the usual benchmark cost; it does not require running
the selected specialists. Do not present selection accuracy as downstream answer
quality or measured cost savings.

### Optional end-to-end experiment

The following steps measure downstream solver quality separately. They are not
prerequisites for the target-selection benchmark requested by the user.

1. Collect representative real requests, authorised for the intended endpoints. Record
   request ID, task, policy constraints, language, source group and request time. Keep
   credentials and identifying fields outside benchmark inputs.
2. Register actual implementations for DETERMINISTIC, SYSTEM_ONE, PREDICTIVE and LLM.
   Record versions, required inputs, language support and deployment eligibility.
   HUMAN_REVIEW is a measured review workflow, not an automatic model output. The
   current scenario's fictional local-endpoint approval does not authorise a new service.
3. Define task-specific output quality before testing: exact checks for rules, independent
   human grading for explanations, and observed outcomes for predictive tasks. Two
   reviewers should adjudicate disagreements without knowing the model identity.
4. Freeze the router and compare it with a fixed baseline on identical requests.
   Run eligible alternative routes on a representative subset to establish whether
   the selected route actually helps. Preserve invalid answers and failed calls.
5. Measure final-answer quality, policy violations, review workload, end-to-end latency
   and total cost: router plus specialist, review/audit/rework and hosting. Report
   paired differences with intervals grouped by independent source unit. Report
   unknown costs as unknown. Route-class accuracy alone is insufficient.

Suggested request table: `request_id, source_group_id, request_time, task, language,
input, policy_constraints, eligible_routes, quality_rubric, split`.
Execution evidence: `request_id, router_version, selected_route, specialist_version,
output, error, router_seconds, specialist_seconds, input_tokens, output_tokens,
priced_cost, currency, reviewer_grade, review_seconds, rework_seconds`.

## Churn experiment

The supplied full parquet contains `ChurnNextMonth`, monthly snapshots and stable
pseudonymised IDs. It is sufficient to start an evaluation against that supplied
target even when the label-generation query is unavailable. State this outcome
definition in the protocol; do not substitute a different definition or infer
retention uplift from it. Use the anonymised file locally, keep target/identifiers
out of predictive features, and quarantine the latest potentially immature months.
The October 2026 local review proposes January 2024–December 2025 for fitting,
January–March 2026 for calibration and April–July 2026 for confirmation. These
boundaries are experimental choices, not verified outcome-maturity guarantees.

Observed formats, ranges and conditional missingness can support declared data
treatment rules. Physical unit conversions require additional evidence; omit
uncertain conversions rather than block statistical prediction. Feed completeness
means whether expected source records arrived, including zero-event periods;
missing aggregates alone cannot establish that. User-supplied completeness rules
can be versioned as experimental policy. No source query or unit dictionary is a
prerequisite to testing these declared policies.

For the supplied support extract, the user confirmed on 7 October 2026 that the
monthly extract is complete and null support fields mean no events. Derived
complaint counts may therefore be zero. With no events, sentiment and event ratios
are not applicable; preserve missing values instead of manufacturing neutral
sentiment or dividing by zero. This source interpretation does not replace the
gold labels in the earlier constructed-policy benchmark.

1. For the supplied-data experiment, use `ChurnNextMonth` unchanged as the target
   and the existing customer/month snapshots. Document that its operational event
   definition and extraction cutoff are unverified; this does not prevent measuring
   prediction against the supplied labels. If a later experiment reconstructs the
   target, agree the event and forecast horizon and collect subsequent outcomes.
   Missing observation follow-up is not a negative churn label. Keep customer IDs
   for grouping only, outside predictive features and model inputs.
2. Check outcome completeness and leakage. Do not include future cancellation,
   account closure, future transactions, churn labels or outcome-derived fields in
   the data-preparation prompts or predictive features.
3. Split by time and customer according to the intended deployment. Fit preprocessing,
   dictionaries, thresholds and predictive models on training/development only.
   Reserve a final untouched confirmation period and independent customers where
   the intended generalisation requires it.
4. Compare the same predictive model and training procedure across original data,
   deterministic preparation and Jev-assisted preparation. Hold feature/outcome
   definitions and evaluation cases constant; preserve an audit of changed fields.
   Include manual preparation only if a measured manual baseline exists.
5. Report ROC-AUC, PR-AUC, calibration and the operating-point precision/recall and
   intervention workload, with paired customer-grouped intervals. Include data
   preparation cost and latency. Predictive improvements are not revenue uplift;
   that needs a separate intervention/control trial and observed business outcomes.

For a future reconstruction, the suggested snapshot table is:
`customer_id, source_group_id, snapshot_date,
feature_* (available at snapshot), observation_end_date, churn_event_date`.
Derive the outcome using the agreed horizon; document censoring, follow-up and
missingness. This reconstruction is optional for the supplied-label experiment.
Do not fabricate outcomes from the current scenario labels.

## Measuring workflow savings

Record representative manual, review, audit and rework times and loaded labour rates.
Audit a sampled accepted subset; measure its error rate independently. Include
specialist/API/hosting and setup costs in one stated currency. Compare total effort
and cost with the measured baseline at the same workload and quality requirement.
Editable report assumptions provide an initial capacity estimate; they do not replace
this measurement or guarantee a payroll reduction.

Before starting an end-to-end operational experiment, choose the actual specialist deployments, outcome
definition, baseline, minimum meaningful improvement, budget and data-sharing scope.
Use these to determine the required sample, rather than copying the scenario counts.

## Questions for business owners

Start with the commercial/customer-retention owner who would act on a prediction:
“Which customer loss are we trying to prevent, how far in advance can we act, and
what action would we take?” Agree the churn definition, seasonality exceptions,
prediction horizon, intervention, missed-loss versus unnecessary-contact trade-off
and success measure. Do not assume a 90-day horizon.

CRM/sales operations and the data owner should confirm the available cancellation,
purchase/delivery and customer-lineage records, observation windows and known data
gaps. Finance should validate customer value, intervention cost and the distinction
between released capacity and realised savings.

For end-to-end routing, the workflow owner and technical lead should identify the actual
specialist services, eligibility constraints and quality reviewers. Ask which
decision or output the routing workflow must deliver, and which errors require
human review. This determines a runnable experiment rather than another route-label
classification test. For the standalone selection test, declared target-role
capabilities and synthetic requests are sufficient; no actual specialist deployment
or execution is required.
