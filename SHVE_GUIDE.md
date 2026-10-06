# SHVE scenario benchmark

This extends the existing model benchmark with six source-grounded scenario tasks.
Underlying records are pseudonymised; activities, corruptions, comparisons and policies
are constructed. The labels are AI-reviewed, not human-approved operational truth.
There are 40 development cases / 20 declared families and 160 evaluation cases /
80 families per task. Related variants and repeats do not increase independent sample
size. Source-customer independence is unverified. No churn model experiment is implied.

## Import and validate

```sh
.venv/bin/python shve.py --archive /path/to/SHVE_Codex_Handoff.zip
```

The archive is preserved under gitignored `data/SHVE_Codex_Handoff`. This command
checks package hashes, all representations, gold/review parity, family separation,
risk labels, nonfinite values and unchanged gold. It loads **only** the combined native
file, never concatenating task/split mirrors. Checks and baseline records go to
`results/shve-preflight`. The supplied 210 repairs are not repeated. Synthetic record
IDs remain a documented shortcut risk; baselines never use them. Original labels and
records remain unchanged.

## Run four models

Configure the existing Jev and GPT keys in the app (include keys when saving locally),
or export `JEV_API_KEY` / `OPENAI_API_KEY` in the process launching the runner. Never
commit keys. Confirm organisational permission to send this particular dataset to both
configured hosted endpoints; scenario `hosted_egress_approved` fields are fictional
policy inputs and do not grant real approval.

```sh
.venv/bin/python shve.py --execute --egress-approved
```

`--egress-approved` records the operator's declaration; it is not a compliance check.
Exactly Jev, the configured GPT model, Nimble and English Laya are selected. CLM stays
installed but is excluded. Two warm-ups and 100 repeated cases (three observations
including primary) produce 1,402 calls/model, 5,608 total. No automatic retries.
Missing credentials are recorded as skipped. Partial results are preserved. Existing
model installations are used offline; no checkpoint downloads. Runtime ownership,
project Nimble port 11435, and system Ollama port 11434 follow `LOCAL_SETUP.md`.

Recover only an incomplete model with the exact saved protocol:

```sh
.venv/bin/python shve.py --execute --egress-approved --recover-from results/RUN_ID --only jev gpt
```

Then use the existing compatible-report combination feature, selecting one complete
source per model. Never rerun a successful paid model to rebuild a report.

## Analyse and share without inference

```sh
.venv/bin/python shve.py --report results/RUN_ID
```

This reads saved evidence, freezes development-only probability thresholds, and
rebuilds the complete offline `report.html`. It makes no network/model calls and does
not modify raw responses, labels or run snapshots. Thresholds and derived decision
metrics are separate analysis files. A threshold is selected on development only:
the lowest candidate threshold with at least ten accepted declared families and no
observed accepted error. If none qualifies, defer all. This small-sample procedure is
provisional, not certified calibration; it does not impose a real risk target. GPT's
configured answer-only adapter does not invent probabilities, so its probability-based
acceptance is unavailable/all-review.

The offline viewer's **Download complete HTML** saves every tab, both splits and all
case evidence in one file. Filters do not restrict that download. SHVE opens on
evaluation results; the **Data split** selector explains that development selects
thresholds and evaluation measures them. They are two partitions of one dataset.
The interface calls them **Threshold selection** and **Performance evaluation**;
no model training occurs. Raw split names remain unchanged in saved evidence.

For the in-app browser, serve a saved report with HTTP attachment downloads:

```sh
.venv/bin/python report_server.py results/RUN_ID --port 8767
```

Open `http://127.0.0.1:8767/report.html`. The server binds only to localhost and
serves only the viewer and its HTML/CSV/JSON downloads. It never calls models or
serves configuration files. This avoids in-app Blob-download failures. Downloaded
HTML remains self-contained for sharing and opening in a standard browser; its
offline download controls use that browser's native Blob support.
See [DOWNSTREAM_EVALUATION.md](DOWNSTREAM_EVALUATION.md) for creating real routing,
workflow-cost and churn evidence that the supplied classification cases cannot provide.
Use [BUSINESS_BENCHMARK_BRIEF.md](BUSINESS_BENCHMARK_BRIEF.md) with the business and
data owners to define the decision, targets and operational confirmation data.
Choice probability acceptance requires a finite returned vector within 0–1,
normalised to 1 within 0.00001, with the selected key present. Malformed or missing
vectors defer to review. This check does not certify calibration.

The run folder also contains `business_summary.md`, `business_decisions.csv`,
`per_task_metrics.csv`, `per_label.csv`, `error_directions.csv`, `risk_curves.csv`,
`baseline.raw.jsonl`, `baseline_summary.csv`, `probability_quality.json` and
`decision_diagnostics.json`. Rule comparisons use paired declared-family intervals
and withhold incremental accuracy for incomplete model cohorts. `analysis_provenance.json`
records fingerprints of the unchanged raw responses, data, configuration and manifest.
Initial configuration, data and request/repeat/batch plans are saved before inference.

In **Results**, expand **Acceptance and workflow assumptions** to explore thresholds
and enter manual/review/audit/rework times, loaded labour cost, monthly volume, setup,
recurring costs and any currency conversion. Audit fraction is 0–1 and applies only to
accepted records. Blank inputs remain unknown. Editing thresholds on evaluation data
is exploratory; use a new confirmation set for later tuning. The same controls exist
in offline **Acceptance and risk** and **Workflow economics** tabs. Existing fallback
replay can also defer on the selected acceptance policy. It is an offline serial cascade
simulation, not deployed workflow measurement. Unknown cascade costs stay unknown.

Label-based coverage preserves the old calculation. New non-review-label coverage
respects task review labels; threshold-qualified coverage also requires a supplied
score. Business-policy-eligible coverage additionally requires explicit task approval.
SHVE policies default to unapproved and entity MATCH is never an automatic merge.
Organisation-declared deployment eligibility is separate from quality and egress.
Score threshold automation is unsupported. Noul uses distinct lower/upper thresholds.

Directional-choice-v1 counts a valid label in `critical_error_choices` as a directional
critical error, with exposed-case denominators. Invalid/API errors remain reliability
failures and fallback events. No exposure means unavailable. Old cases retain legacy
scoring. Confusion-derived error tables retain all other wrong-answer directions.
Multiclass Brier uses valid complete normalized vectors only (sum of squared errors,
range 0–2); reliability bins are descriptive, not proof of production calibration.

Deterministic baselines are shown separately. Numeric checks, authoritative identity
matching and readiness rules directly implement the constructed rubric. The router
uses transparent keywords; sector/label rules use development vocabulary and category
meanings, never row IDs or evaluation labels. In-process rule timings are not HTTP
latencies. Limited templates constrain generalisation. Vision scores do not measure
execution savings or downstream model quality. Economic scenarios extrapolate the
constructed cohort and release capacity; they do not guarantee payroll savings.

Use the per-task decision table first. Case-pooled accuracy weights cases; a macro
average across tasks gives each task equal weight. Missing risk/latency/consistency
business targets block relevant pilot decisions. Small samples and pending human
review leave production recommendations inconclusive regardless of accuracy.

## Historical retrieval adjudication

In the 2 October generic run, ten Jev retrieval errors share
`synthetic:retrieval:pattern-7`. Asking where to email support while providing only
“support is available by email” leaves relevance versus answerability ambiguous.
Historical labels/scores are preserved. This requires human rubric adjudication;
it is not evidence that Jev's NONE is necessarily wrong or that C should be relabelled.
This old synthetic screening cohort is separate from SHVE. No future fixture is
silently substituted into old evidence. Ineffective historical option-order pairs
remain excluded; absence of an effective pair is not a passed robustness test.
