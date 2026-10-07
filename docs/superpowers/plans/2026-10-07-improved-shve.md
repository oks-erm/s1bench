# Improved SHVE Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for orchestration and verification; independent case, metric and report work is delegated under dispatching-parallel-agents.

**Goal:** Execute a frozen improved four-model benchmark and deliver one complete report with the existing categories and metrics.

**Architecture:** Retain the existing benchmark runner, adapters, locks, raw evidence and offline viewer. Separate authored semantic fixtures, local parquet sampling, weighted churn analysis and improved-run orchestration into focused modules. Preserve the earlier handoff runner and saved runs.

**Tech Stack:** Python 3.11, standard library inference/analysis, optional PyArrow preparation, existing HTML/JavaScript viewer.

**Spec:** User instructions in this chat: retain six categories (rename model_routing), add substantive contextual cases, use supplied churn outcomes, exclude CLM/CLEF, compare Jev/GPT/Nimble/Laya, preserve all applicable report metrics and downloads.

## Global Constraints

- Work on main; no push; never commit keys, customer data, results or weights.
- Four models only, one local model at a time; preserve completed evidence.
- Freeze data, policies, repeat plans and historical prior before inference; no evaluation tuning.
- Hosted transfer of new source features requires explicit approval.
- No claims of measured downstream answer quality, savings or retention uplift.

## Review Focus

- Anonymous features must not expose IDs, outcome values or future closure flags.
- Oversampling must not inflate population precision or headline accuracy.
- Missing or invalid predictions must remain failures/abstentions with visible coverage.
- Recovery must reject complete models and changed data/settings.
- Older reports and strict deterministic-label guards must remain compatible.

### Task 1: Frozen inputs and analysis

- [x] Implement and test 360 authored semantic cases, including model_routing, with input-only lexical controls.
- [x] Implement and test 1,000 source churn cases with month/label sampling weights and prior ending November 2025.
- [x] Implement weighted prediction metrics and development-only threshold selection; test ties, failures and split separation.
- [x] Add explicit observed-outcome validation and opt-in strict GPT response schema; retain default behavior.

### Task 2: Improved runner and report

Files: shve_improved.py, tests/test_shve_improved.py, report_export.py, report_assets/report.js, tests/test_improved_report.py.

- [x] Test and implement run_config(existing, cases), prepare_suite(folder), baseline_records(cases, cfg), churn_summaries(records, cases, cfg), export_analysis(folder).
- [x] Freeze 1,360 primary cases, two warm-ups and 200 repeats per model: 6,248 requests across four models.
- [x] Reuse benchmark.run and combine_reports; verify recovery rejects complete models or changed frozen inputs.
- [x] Embed weighted churn quality and confusion alongside all existing tabs, metrics, colours and downloads; label unweighted churn tables as sample diagnostics.
- [x] Verify offline payload/export behavior and localhost downloads.

### Task 3: Execution and delivery

- [x] Update README, SHVE_GUIDE, optional data dependencies, CI syntax and AGENTS.
- [x] Run full tests and syntax checks; independent review; fix important findings before inference.
- [x] Execute local models sequentially and hosted models once approved; recover incomplete models only.
- [x] Verify request counts, errors, model identities and frozen evidence; point owned report server at the combined run.
- [x] Explain observed winners and uncertainty using the latest complete evidence; focused local commit; no push.

## Completion verification — 7 October 2026

- 137 full tests passed; Python, browser scripts and launcher syntax passed.
- Four complete model protocols: 6,248 unique requests, no API errors or invalid answers.
- Laya startup was recovered after restoring existing macOS cloud placeholders; no completed hosted or Nimble requests were repeated.
- Frozen evidence and grading audits passed for the combined comparison. The interrupted original export and its checksum provenance are preserved separately.
- All 14 viewer sections rendered; 56 confusion cohorts and weighted churn analysis are present.
- HTML/CSV/JSON HTTP attachments returned 200; all three browser downloads succeeded. Downloaded HTML matches the complete export byte for byte and contains no configured keys.
- Observed leaders are descriptive. Core tasks have 24 evaluation premise families; churn intervals are withheld because customer clusters cross sampling strata. Neither retrospective discrimination nor routing target selection establishes measured business uplift.
