"""Complete offline report viewer. Standard library only; never invokes models."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "report_assets"
MODEL_COLORS = {"jev": "#0072B2", "laya": "#E69F00", "nimble": "#009E73",
                "gpt": "#8064C9", "clm": "#D55E00"}


def build_payload(report, *, analysis_cfg=None, current_prices=None, analysis=None,
                  price_basis="Prices saved with run"):
    import benchmark as b
    cfg = copy.deepcopy(analysis_cfg or report["config"])
    args = (report["records"], report["cases"], cfg, report.get("availability"), current_prices,
            report["manifest"].get("stability_sample_ids"))
    summaries, records, profiles, quotes, stability, robustness = analysis or b.summarize(*args)
    baseline = cfg.get("business", b.default_config()["business"])["baseline"]
    # Reuse the Python implementation for confidence intervals and recommendations.
    # Every reference is computed at export time; the viewer never approximates these.
    by_reference = {baseline: summaries}
    for alias in profiles:
        if alias != baseline:
            variant = copy.deepcopy(cfg)
            variant.setdefault("business", {})["baseline"] = alias
            by_reference[alias] = b.summarize(args[0], args[1], variant, *args[3:])[0]
    public_quotes = b.scrub_config({"models": list(quotes.values())})["models"]
    # Do not embed connection credentials, custom headers or repeated model profiles.
    for quote in public_quotes:
        for key in ("headers", "api_key_env"):
            quote.pop(key, None)
    manifest = copy.deepcopy(report["manifest"])
    manifest.pop("quotes", None)
    manifest["analysis_business_targets"] = cfg.get("business", {})
    manifest["price_basis"] = price_basis
    manifest["analysis_quotes"] = public_quotes
    manifest["original_run_costs"] = report["manifest"].get("analysis_costs", [])
    manifest["analysis_costs"] = b.cost_totals(records, quotes)
    from decision_analysis import diagnostics, risk_curve, policy_for, baseline_comparisons, cascade_curves
    decision = {"diagnostics": diagnostics(report["cases"], records), "risk_curves": [],
                "workflow_cost": cfg.get("workflow_cost", {}), "acceptance": cfg.get("acceptance", {}),
                "tasks": cfg.get("tasks", {}), "frozen_acceptance": cfg.get("frozen_acceptance", {})}
    for summary in summaries:
        cohort_rows = [r for r in records if b.primary_row(r) and
                       (r["model"], r["dataset"], r["task"]) == (summary["model"], summary["dataset"], summary["task"])]
        curves=risk_curve(cohort_rows,policy_for(cfg,summary['task'],summary['model']),summary['planned'])
        decision['risk_curves'] += [{'model':summary['model'],'dataset':summary['dataset'],'task':summary['task'],**curve}
                                   for curve in cascade_curves(curves,records,summaries,summary,baseline)]
    study_design = None
    improved = cfg.get('shve_protocol', {}).get('suite') == 'improved'
    default_dataset = ('shve_improved_evaluation' if improved else
                       'shve_v2_evaluation' if cfg.get('shve_protocol') else '')
    if cfg.get("shve_protocol"):
        study_design = {
            "split_counts": {role: sum(c.get("split") == role for c in report["cases"])
                             for role in ("development", "evaluation")},
            "development_purpose": "Choose and freeze rules and probability thresholds; excluded from evaluation scores.",
            "evaluation_purpose": "Measure the frozen approach on separate cases; both splits belong to one SHVE dataset.",
            "downstream_requirements": {
                "routing_selection": "Test whether the model chooses the target specified by a declared topic-and-complexity policy. Target models need not be called for this selection test.",
                "routing_quality": "Measuring downstream answer quality additionally requires executing the selected specialists and grading their outputs; this is separate from target-selection accuracy.",
                "workflow_savings": "Measure manual, review, audit and rework time, hosting and specialist costs. Editable cost scenarios are estimates, not measured savings.",
                "churn_uplift": "A separate labelled monthly dataset can measure prediction quality against its supplied churn outcomes. Measuring retention uplift additionally requires an intervention/control trial; prediction accuracy alone does not establish uplift."},
            "additional_evidence": "This benchmark measures accuracy against explicit supplied rules; defining test rules does not require operational business approval. Operational adoption separately requires validated labels, source-entity lineage and approval for the actions taken. Changed rules should be tested as a new policy version, preserving the earlier results."}
        if improved:
            from shve_improved import baseline_records, BASELINE_VERSION, churn_summaries
            study_design.update(
                prediction_scope=cfg['shve_protocol'].get('prediction_scope'),
                churn_protocol=cfg['shve_protocol'].get('churn_protocol'),
                threshold_rule=cfg['shve_protocol'].get('threshold_rule'),
                additional_evidence='Authored semantic cases test the declared experimental policies. Monthly churn cases test supplied observed outcomes with sampling weights, conditional on feature availability. These results do not independently verify prospective forecasting, retention uplift, operational safety or business approval.')
            decision['churn_prediction'] = churn_summaries(records, report['cases'], cfg)
            churn_index = {(entry['model'], entry['dataset'], entry['task']): entry
                           for entry in decision['churn_prediction']}
            # Supplied analysis can be reused by the caller; annotations belong to
            # the export only and must not mutate the original summary evidence.
            by_reference = copy.deepcopy(by_reference)
            summaries = by_reference[baseline]
            for reference_summaries in by_reference.values():
                for summary in reference_summaries:
                    entry = churn_index.get((summary['model'], summary['dataset'], summary['task']))
                    if entry:
                        summary.update(quality_basis='oversampled diagnostic; population metrics are weighted',
                                       churn_predictive_fixed=entry['fixed'],
                                       churn_predictive_selected=entry['selected'],
                                       churn_threshold_selection=entry['threshold_selection'])
        else:
            from shve import baseline_records, BASELINE_VERSION
        baseline_rows = baseline_records(report["cases"], cfg)
        rule_cfg = copy.deepcopy(cfg)
        rule_cfg["models"] = [{"name":"rules", "display_name":"Deterministic rules", "enabled":True,
                               "api":"rules", "model":BASELINE_VERSION, "endpoint":"", "deployment":"local"}]
        decision["baseline_records"] = baseline_rows
        decision["baseline_summaries"] = b.summarize(baseline_rows, report["cases"], rule_cfg, bootstrap=300)[0]
        comparison_summaries = [r for r in summaries if not improved or r['task'] != 'churn_prediction']
        decision["baseline_comparisons"] = baseline_comparisons(comparison_summaries,records,decision["baseline_summaries"],baseline_rows)
        decision["baseline_basis"] = ("Authored semantic tasks use a frozen simple control; beating it does not establish a general advantage over code. Churn uses a historical probability baseline frozen before confirmation. Latency is local function time, not comparable HTTP latency. No paid API cost; hardware cost unknown." if improved else
            "Input-only deterministic rules; development vocabulary frozen before evaluation. Latency is local function time, not comparable HTTP latency. No paid API cost; hardware cost unknown.")
        rule_tasks = {"semantic_validation", "entity_match", "vision_routing", "data_gap_identification"}
        decision["baseline_interpretations"] = [
            {"dataset": row["dataset"], "task": row["task"],
             "interpretation": "Those results currently demonstrate policy compliance; they do not demonstrate an advantage over straightforward code.",
             "rules_success_rate": row["success_rate"]}
            for row in decision["baseline_summaries"]
            if row["task"] in rule_tasks and row["status"] == "complete" and row["success_rate"] == 1]
        warnings_shve = ["Model routing measures target selection; downstream answer quality, savings and churn uplift were not measured." if improved else
                         "Vision is policy classification only; downstream routing quality, savings and churn uplift were not measured."]
        manifest.setdefault("analysis_warnings", []).extend(warnings_shve)
    # Investigation notes remain in the audit artifacts, not in shared reports.
    warnings = list(dict.fromkeys(w for w in manifest.get("analysis_warnings", [])
                                  if not w.startswith(("Option-order robustness is unavailable", "CLM diagnostic:"))))
    manifest["analysis_warnings"] = warnings
    return {"format_version": 1, "exported_utc": b.utc(), "run_name": Path(report.get("folder", "report")).name,
            "baseline": baseline, "summaries_by_reference": by_reference, "profiles": profiles,
            "quotes": {q["name"]: q for q in public_quotes}, "manifest": manifest,
            "cost_totals": manifest["analysis_costs"], "warnings": warnings,
            "stability": stability, "robustness": robustness,
            "confusion": b.confusion_tables(records), "cases": report["cases"],
            "records": [{k: v for k, v in r.items() if k != "profile"} for r in records],
            "business": cfg.get("business", {}), "colors": {k:v for k,v in MODEL_COLORS.items() if k in profiles},
            "study_design": study_design, "default_dataset": default_dataset,
            "decision_analysis": decision, "metric_guide": METRIC_GUIDE, "protocol": PROTOCOL_TEXT if "clm" in profiles else PROTOCOL_TEXT.replace(
                "CLM embedding caches affect timings; record server cache/context settings.\n", "") }


def render_report(payload):
    # JSON in a script element must not permit an input containing </script> to
    # terminate that element. All dataset values are rendered with textContent.
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    css = (ASSETS / "report.css").read_text("utf-8")
    logic = (ASSETS / "logic.js").read_text("utf-8")
    script = (ASSETS / "report.js").read_text("utf-8")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; "
            "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; "
            "connect-src 'none'; base-uri 'none'; form-action 'none'\">"
            "<title>System 1 benchmark — complete report</title><style>" + css + "</style></head><body>"
            "<header><p class='eyebrow'>SYSTEM 1 BENCHMARK · COMPLETE REPORT</p>"
            "<h1>System 1 benchmark</h1><p id='subtitle'></p>"
            "<p class='muted'>Explore this saved run offline. No installation, API keys or model calls are needed.</p></header>"
            "<main><div id='notices'></div><div id='summary-cards' class='cards'></div>"
            "<div id='filters' class='filters'></div><nav id='tabs' role='tablist' aria-label='Report sections'></nav>"
            "<section id='content' role='tabpanel' tabindex='0'></section>"
            "<footer id='downloads'></footer></main><noscript>Enable JavaScript to explore the embedded report data.</noscript>"
            "<script id='report-data' type='application/json'>" + data + "</script>"
            "<script>" + logic + "</script><script>" + script + "</script></body></html>")


def complete_html_report(report, **kwargs):
    return render_report(build_payload(report, **kwargs))


PROTOCOL_TEXT = """
**Fair comparison.** Models receive the same state, question and rubric.
Native decision models receive typed requests; chat models return the same
decision in JSON. Adapters render content differently, so results describe
the configured deployment. Inspect captured IDs, parameters and effort.

**Quality.** Each primary case counts once. Partial-run success divides by all planned cases; unattempted cases are not successes. Sampling intervals are withheld for partial cohorts. API failures and invalid answers
count as failures for task success. Choice uses exact labels, noul uses the
saved decision threshold, and scores use the saved tolerance. F1 balances
represented gold classes. Brier and MAE use valid numeric outputs; read
validity coverage alongside them.

**Evidence.** Compare matched cases in the same dataset/use case. Family
bootstrap intervals keep related variants together. Wilson intervals assume
one independent case per family. Grouped boundary outcomes cannot estimate
unseen errors. Intervals are unadjusted 95% intervals conditional on this
sample; selecting among many models needs confirmation on a fresh holdout.

**Batches and stability.** Ten batches organize one dataset; they are not ten
independent datasets. Primary outcomes are pooled. F1 and p95 are not averaged
across batches. Exact repetitions measure consistency without increasing the
independent accuracy sample size. A stable wrong answer is still wrong.
Paraphrase, distractor, option-order and changed-fact checks are measured only
where explicit labelled pair metadata exists.

**Latency.** Median/p95 show completed HTTP calls without API errors, including
invalid generated answers. Attempt p95 includes failures/timeouts. Warmups are
excluded from primary quality. Client time includes network and parsing.
This sequential test does not establish production concurrency or throughput.
CLM embedding caches affect timings; record server cache/context settings.

**Cost.** Saved token usage is repriced under the selected rates. Missing paid
usage remains unknown. Local API fees are zero; hosting is separate. Run spend
includes warmups, connection checks and repetitions. Per-1k prices with partial
coverage describe known calls only. Forecasts require full cost coverage.
Hidden reasoning tokens already included in total output usage are not added
again. Estimates and 30-day hosting scenarios are not billing invoices.

**Decision.** Inspect unsafe errors, critical failures, uncertainty, latency,
consistency and cost together. The pilot recommendation requires reviewed
real holdout labels and declared business targets. It does not certify safety.
Synthetic cases and AI draft labels are screening evidence.
"""

METRIC_GUIDE = [('Label-based automation coverage',
  'Historical auto_coverage counts valid Choice decisions outside configured safe labels; no confidence threshold. New non-review-label coverage also respects task review labels such as HUMAN_REVIEW.'),
 ('Threshold-qualified coverage',
  'Accepted / all planned primary cases under explicitly chosen probability or vendor-confidence thresholds. Missing scores defer to review; Score automation is unsupported. Accepted accuracy uses accepted cases only; show failures and coverage alongside it.'),
 ('Business-policy-eligible coverage',
  'Threshold/label acceptance plus organisation-approved task eligibility. Deployment approval is separate. AI-reviewed fixtures cannot certify live automation; entity MATCH cannot trigger automatic merging.'),
 ('Workflow cost scenario',
  'Manual baseline minus review, audit of accepted cases, and estimated rework. Labour, volume, hosting and FX are assumptions; error extrapolations use the evaluated cohort, not true operational prevalence. Released capacity is not guaranteed cash savings.'),
 ('Directional critical risk',
  'For directional-choice-v1, a valid predicted label in the case critical_error_choices is an error; exposure is a case with a nonempty risk list. Failures remain reliability failures. No exposure is unavailable, not safe. Legacy cases retain historical scoring.'),
 ('Task success',
  'Correct / all planned cases; errors and invalid answers count as failures. Use for the '
  'production outcome, only compare complete cohorts.'),
 ('95% interval',
  'Sampling uncertainty at independent-family level. Related conversations are grouped. A narrow '
  'interval needs diverse reviewed data; repetitions do not enlarge the independent sample.'),
 ('Paired gap vs reference',
  'Difference on exactly the same cases with a paired family interval. A replacement is plausible '
  'only when the lower bound clears your allowed quality drop.'),
 ('Macro F1',
  'Equal weight to classes present in the gold set. Reveals minority-class weakness hidden by '
  'accuracy; inspect missing labels in your dataset.'),
 ('NONE recall / precision',
  'Recall: catches out-of-scope requests. Precision: avoids unnecessarily refusing useful work. '
  'Non-NONE on NONE shows harmful forced routing.'),
 ('Risk exposure and harmful errors',
  'Denominator depends on the failure: out-of-scope messages, positive PII cases, unauthorized '
  'actions, or high-escalation gold. Set the direction per task. Grouped or small samples may not '
  'establish your risk bound.'),
 ('Critical failures',
  'Wrong, missing or invalid answers on cases tagged critical. Review the cases before automation; '
  'one severity label cannot encode every business impact.'),
 ('Brier score',
  'Mean squared error of binary probabilities on valid answers, lower is better. Valid for '
  "probabilistic yes/no forecasts; not a proof that different providers' confidence fields are "
  'comparable.'),
 ('MAE / score tolerance',
  'Mean absolute score error plus success inside your configured rubric tolerance. Scores are '
  'zero-based rubric indices; MAE alone can hide dangerous under-escalation.'),
 ('Valid answers / API errors',
  'Separates malformed decisions from HTTP/network failures. Reliability includes both; '
  'completed-answer quality alone can flatter an unreliable endpoint.'),
 ('p50 and p95 latency',
  'Client wall time on completed requests after warm-up. Shows typical and tail delay, including '
  'transport/response parsing. Attempt p95 also includes failed requests. Serial testing is not '
  'concurrent production capacity.'),
 ('Repeat agreement',
  'Same input and settings, repeated in separate rounds; equality for categories/binary decisions '
  'and configured score tolerance. Consistency can be consistently wrong, so inspect '
  'all-repeats-correct and critical flips.'),
 ('Robustness pair success',
  'All linked equivalent/changed-fact cases correct. Tests distractors, option order, paraphrase '
  'and sensitivity to changed facts; report alongside repeat consistency.'),
 ('API cost / 1k and per correct',
  'Observed token charges at recorded or current pinned rates. Unknown usage/prices stay unknown; '
  'local API charges are 0. Cost per correct includes failed attempts when all charges are known.'),
 ('Hosting and monthly scenarios',
  'Explicit hourly hosting and volume assumptions. Active serial wall time and 720-hour '
  'reservation are separate scenarios; throughput/utilization must be measured before a fleet '
  'budget.'),
 ('Fallback replay',
  'Offline replay: invalid/error or selected fallback labels trigger the reference model. Adds '
  'both costs and serial latencies. Valid only on complete shared cases; real agent behavior can '
  'differ.'),
 ('Ten batches',
  'Whole-family balanced partitions to spot instability over time and mix. They reuse one dataset, '
  'so their average does not provide ten independent experiments.')]
