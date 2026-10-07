# Working on this checkout

Continue development on `main` as explicitly requested by the user. Do not push unless asked.

## Purpose and architecture

System 1 benchmark compares typed choice, yes/no, and rubric decisions across model
deployments. `benchmark.py` is the standard-library CLI, adapters, scoring, and saved
evidence layer. `bench_ui.py` is the Streamlit dashboard. `starter_data.py` creates
synthetic fixtures; `data_prompt.py` supplies data preparation guidance. There is no
database or migration system. Local config and JSONL data flow into HTTP model calls,
then frozen configuration, raw responses, and summaries are saved under `results/`.

`local_models.py` and `Start Local.command` launch one installed local model at a time
on this 16 GB Apple Silicon Mac. `local_clm_server.py` supplies localhost HTTP transport
around the explicitly approved community CLM MLX port. Model code/weights and isolated
environments live under gitignored `.venv/local/`. See `LOCAL_SETUP.md` for exact profiles,
ports, provenance, limitations, fresh installation, startup, recovery, and smoke-test instructions.
`requirements-laya.txt` and `requirements-clm.txt` pin the tested inference dependencies;
`config.mac.example.json` contains public model settings with all models disabled.
Never generate that example by copying secrets or private settings from config.json.

`local_runtime.py` owns automatic server startup/shutdown for installed Mac profiles.
The dashboard passes its context manager to `benchmark.run`, which executes models
sequentially in one shared report. The CLI opts in with `--manage-local`. Default CLI
execution remains interleaved. Never stop a server this context did not start. Keep
identical cases, batch assignments and repeat samples across models; record execution
order in the manifest. Every completed/partial run also exports `report.html`.
A file lock prevents concurrent benchmarks in the same results directory across tabs
and processes. It releases on completion or process exit; never bypass an active lock.
`benchmark.combine_reports` combines explicitly chosen complete model runs without
new calls. It checks source evidence, case/label equality, scoring settings, batches
and repeat coverage, rejects duplicate models, and records source fingerprints.
Use it to recover one failed local model without rerunning paid providers. Preserve
partial source reports; never silently replace model records or mix incompatible runs.
CLM caps MLX's free-buffer cache at 256 MiB; its separate projection cache is unchanged.
Pricing scenarios preserve saved rates when current fields are blank. Current rates
must match both model and endpoint; explicit zero prices are valid. The dated Jev
list-price fallback applies only to direct TypeSafe responses identifying Jev 1.13.0.
Keep its official source/version/date visible. Never reuse that price for unknown
versions or gateways. Repricing changes analysis exports, not frozen raw evidence.
Use input_json (order-preserving) for requests/chat payloads and jsonl_bytes for
snapshots. canonical sorts keys only for order-independent comparisons/hashes.
Combining runs must check the request_serialization policy and ordered inputs;
missing policy means legacy sorted_keys. Do not count identical saved option orders
as an option-order robustness test. Keep old evidence unchanged. CLM_DIAGNOSTIC.md
records the local port investigation and its limits. CLM and option-order investigation
notices are omitted from shared reports as requested; retain unrelated analysis warnings.
`report_export.py` and `report_assets/` build a single-file offline viewer, embedding
all datasets, cases and response evidence. Protocol text and the metric guide are
shared with Streamlit. Reference comparisons use Python's existing summaries;
browser fallback replay in logic.js must match benchmark.fallback_replay (Node parity
tests run when Node is available). Never fetch CDNs or call model APIs from exports.
The viewer offers a complete HTML download, clearing generated DOM before saving so
reopening does not duplicate controls. SHVE defaults to evaluation; development and
evaluation are partitions of one dataset, never pooled into a claimed holdout score.
`report_server.py` serves one saved viewer and allowlisted HTTP attachments on
127.0.0.1 for in-app browsers that cannot save Blob URLs. It never exposes config
files or runs inference. Offline files retain standard-browser download support.
Do not change Python analysis APIs during a running benchmark: loaded modules can
conflict with lazily imported exporters. Export failures must be recovered in a fresh
process from saved responses, without repeating completed hosted calls.
Escape embedded JSON against script termination and render dataset text with
textContent. Do not include connection credentials or custom request headers.
Export all cohorts regardless of current dashboard filters. Prices/business targets
are captured settings; only volume/fallback scenarios recalculate in the browser.

## Commands and validation

- Python 3.10+; this checkout uses `.venv/bin/python` (3.11).
- Standard setup/dashboard: `python3 start.py` (creates `.venv`, installs requirements).
- This Mac: `.venv/bin/python local_models.py laya` (or `nimble`, `clm`).
- Tests: `.venv/bin/python -m unittest discover -s tests -v`.
- Syntax: `.venv/bin/python -m py_compile benchmark.py bench_ui.py report_export.py report_server.py decision_analysis.py decision_ui.py shve.py starter_data.py data_prompt.py start.py local_models.py local_runtime.py local_clm_server.py`.
- Browser script syntax: `node --check report_assets/logic.js` and `node --check report_assets/report.js`.
- Launcher syntax: `zsh -n 'Start Local.command'`.
- No separate build, linter, or typechecker is configured.
- `.streamlit/config.toml` uses viewer toolbar mode to hide Streamlit's publishing
  controls. Model selection belongs on Run; Models & config contains connection settings.
- CI `.github/workflows/checks.yml` runs Python 3.11 syntax/tests on Linux, Windows and macOS
  for main and codex/system-one-benchmark. No deployment is configured there.
- Before finishing: inspect diff/status, run affected tests and syntax checks, verify
  actual localhost health and inference if runtime behavior changed. Tests use mocks;
  model smoke tests are separate and require installed weights. Do not equate CLI exit
  success with successful inference: inspect availability and record-level errors.

## Safety and constraints

- Respond in English. Preserve user changes; check branch/status before editing.
- Do not push, deploy, delete data/weights, or change hosted integrations without authorization.
- Never commit secrets, `config.json`, `.env`, data, results, environments, or weights.
- Local model profiles need no keys. Hosted profiles optionally use `JEV_API_KEY` and
  `OPENAI_API_KEY`; keep them disabled unless specifically requested.
- Bind local servers to 127.0.0.1. Do not broaden access or weaken protections.
- Preserve the existing system Ollama on port 11434; project Nimble uses port 11435.
- Run one local model at a time on this 16 GB Mac. Stop an active run before switching;
  choose benchmark participants on Run. Starting a server preserves saved selections.
- Clearly label the CLM community MLX 8-bit variant; never present it as exact reference
  CLM. Preserve quantization, revision, truncation and cache provenance in reports.
- Laya is configured for English; do not infer multilingual support.
- Synthetic fixtures and a three-case smoke test do not prove real-world quality.
- Keep changes focused. Update tests for behavior changes and these notes when the
  architecture, startup workflow, configuration, or serving assumptions change.

## SHVE extension (October 2026)

`decision_analysis.py` owns pure acceptance/risk/economics/diagnostics calculations;
`decision_ui.py` supplies Streamlit controls. `report_assets/logic.js` mirrors acceptance,
economics and cascade logic; test parity before changing formulas. `shve.py` validates
handoff archives, runs input-only rule baselines and an explicit four-model SHVE protocol.
See `SHVE_GUIDE.md`. Data/results remain ignored. Do not use evaluation responses to tune
thresholds or baselines; frozen threshold analysis is separate from raw evidence.
Directional risk is opt-in via case `critical_error_choices`; legacy scoring stays intact.
Organisation approval, scenario eligibility, and model quality are distinct. No Score
threshold automation; missing probabilities defer. Fully unknown costs cannot imply savings.
SHVE `HUMAN_REVIEW` is excluded from all label-coverage measures; legacy safe-label
defaults remain unchanged. Frozen `accept_none` affects threshold acceptance only,
never semantic review labels or review recall. Rule comparisons require complete,
matched primary cohorts and use the existing paired-family intervals.
The runner saves initial inputs/configuration and batch/repeat plans before inference.
Analysis exports verify that raw/config/data/manifest fingerprints remain unchanged.
Include `decision_analysis.py decision_ui.py shve.py` in Python syntax validation.
SHVE is 1,402 calls per selected model (four-model total 5,608); explicit recovery
may select incomplete models only and must retain the original protocol/inputs.

`shve_improved.py` is a separate revised protocol, not a change to `shve.py`:
`shve_cases.py` supplies 360 authored semantic cases in six categories (including
`model_routing`); `churn_data.py` locally prepares 1,000 weighted source snapshots.
The new plan is 1,562 calls/model / 6,248 total. Optional parquet preparation uses
`requirements-data.txt`; inference/analysis remain standard library. Include these
three modules and `churn_analysis.py` in syntax checks. Default source data/results
are ignored, and hosted transfer of source features requires explicit approval.
No IDs, outcome values, future contract flags, weights or label reasons may enter
requests. Null monthly support counts mean no events under the confirmed complete
extract; sentiment/ratios remain NA. Unit conversions cannot be guessed.
`churn_analysis.py` owns N/n-weighted probability metrics and calibration-only
experimental threshold selection. Keep fixed 0.5 classification separate from it;
never claim unweighted oversampled churn accuracy as operational performance.
Observed stochastic binary labels require explicit task `label_semantics:
observed_outcome`; default deterministic duplicate/gold guards remain active.
Source as-of joins and generating outcome SQL are unverified; results describe
retrospective supplied-label discrimination, not certified forecasting or uplift.
Lexical controls are deliberately limited; neural advantage over them is not proof
of advantage over all code. Strict GPT schema is opt-in on Responses endpoints.
Improved exports retain every existing report tab, metric, colour and download.
Complete model recovery is rejected. Audit saved evidence before analysis and never
edit lazy-imported analysis APIs during an active run.
`shve_recovery.py` resumes only missing warm-up/primary/repeat slots; it preserves
prior attempts and source fingerprints. Failed warm-ups relabelled connection
checks are counted as already issued slots, with source_phase retained. Approval
declarations belong to request_plan.json provenance, not scoring configuration,
so otherwise compatible local and hosted sources can combine. Add this module to
syntax validation. Never retry an already present slot to improve scores.
