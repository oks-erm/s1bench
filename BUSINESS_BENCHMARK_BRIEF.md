# Inputs for a benchmark that supports a business decision

The current SHVE run compares configured Jev, GPT, Nimble and Laya on six supplied
scenario tasks, with deterministic baselines and separate development/evaluation
scores. It can screen where a controlled pilot is worth investigating. It does not
substitute for human-validated operational labels or measured downstream outcomes.

Explicit test policies can be defined without knowing the organisation's current
business rules. Accuracy measures whether models follow those declared policies;
operational approval is a separate question about using their decisions in practice.
When rules change, version the policy, prompts, cases and expected answers, then
evaluate the new version separately. Preserve earlier evidence rather than changing
its labels or comparing scores as though the underlying task stayed identical.

## First meeting: business/workflow owner

Agree these seven answers before designing the next confirmation experiment:

1. Which workflow and decision must the benchmark support? Choose the model or
   rules for a pilot, approve a particular automation step, or test predictive uplift.
2. What output is required, and when should the system ask for human review?
3. Which mistakes are unacceptable? Give concrete examples and acceptable rates
   for each direction, rather than one overall accuracy target.
4. What improvement over the current workflow would justify a change?
5. How many records occur monthly, in which languages/business units, and which
   uncommon cases matter? Obtain observed workload proportions.
6. What are the latency and cost limits, including review, audit and rework?
7. Who owns the decision and can approve a controlled pilot?

For churn, the commercial/retention owner must additionally define the loss event,
prediction horizon, intervention and intended value. A 90-day horizon is an example,
not a default. Finance validates customer value and intervention/workflow costs.

## Evidence request and responsibilities

| Input | Owner | Engineering work |
|---|---|---|
| Representative real cases, permitted languages and business units | Data owner + domain expert | Validate inputs, leakage, missingness and subgroup coverage |
| Stable source/customer lineage and observation dates | Data owner | Group related records; create appropriate customer/time splits |
| Rubric and independently reviewed correct answers | Domain reviewers | Prepare blinded reviews, adjudication queue and versioned gold |
| Current rules/manual process and its measured performance | Workflow owner | Implement reproducible baselines and paired comparisons |
| Error/review/latency/cost targets | Business owner + Finance | Translate agreed targets into explicit decision gates |
| Manual/review/audit/rework timings and hosting/API rates | Operations + Finance + technical lead | Compute measured cost/workload comparisons |
| Actual eligible specialist deployments and final-output quality labels | Technical lead + workflow owner | Execute end-to-end routing comparison |
| Dated features and observed churn outcomes | Retention owner + CRM/data team | Compare predictive pipelines before/after data preparation |

The business team supplies evidence and approves definitions; it does not need to
write benchmark code. Engineering can prepare the files, validate them, run approved
experiments and produce the report. Human adjudication and organisational approvals
cannot be replaced by a model's own judgement.

## Practical sequence

1. Finish the current four-model scenario run and preserve its evidence unchanged.
2. Prioritise sector/label stewardship: obtain unseen, representative cases and human
   labels; compare with stronger dictionary/reference methods and current review.
3. Define targets and sample size around the minimum useful improvement and required
   precision for important errors. Do not use a universal case-count gate or pad with
   duplicate templates. Include sufficient exposure to rare critical directions.
4. Freeze prompts, vocabulary and thresholds on development data. Keep a separate
   confirmation set with verified lineage; do not tune after inspecting its errors.
   Validate equivalent output contracts across adapters on development examples,
   including exact choice keys and supported schema constraints. Separate reasoning
   errors from interface/format failures; never repair evaluation answers post hoc.
5. Run paired quality, risk, consistency, language/subgroup, latency and total-cost
   comparisons. Report missing evidence and inconclusive intervals explicitly.
6. Select rules where they solve the task adequately; select a model-assisted pilot
   only where observed incremental benefit and operational constraints justify it.
7. Execute routing and predictive experiments separately when their inputs exist.
   See [DOWNSTREAM_EVALUATION.md](DOWNSTREAM_EVALUATION.md) for tables and protocol.
8. Validate a selected workflow in shadow mode, then a controlled pilot, before making
   broad automation or realised-savings claims. A pilot needs its own owner and scope.

Development and evaluation are partitions of one dataset. Development selects the
approach; evaluation measures that frozen approach. Removing the separation would
make tuned results harder to trust.

The test design follows the emphasis on context-relevant measurement and representative
data in the [NIST AI RMF Measure guidance](https://airc.nist.gov/airmf-resources/playbook/measure/).
Customer/time splits should match deployment and avoid the dependence/leakage pitfalls
covered by [scikit-learn's cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).
