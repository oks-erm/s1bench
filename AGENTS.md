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
Escape embedded JSON against script termination and render dataset text with
textContent. Do not include connection credentials or custom request headers.
Export all cohorts regardless of current dashboard filters. Prices/business targets
are captured settings; only volume/fallback scenarios recalculate in the browser.

## Commands and validation

- Python 3.10+; this checkout uses `.venv/bin/python` (3.11).
- Standard setup/dashboard: `python3 start.py` (creates `.venv`, installs requirements).
- This Mac: `.venv/bin/python local_models.py laya` (or `nimble`, `clm`).
- Tests: `.venv/bin/python -m unittest discover -s tests -v`.
- Syntax: `.venv/bin/python -m py_compile benchmark.py bench_ui.py report_export.py starter_data.py data_prompt.py start.py local_models.py local_runtime.py local_clm_server.py`.
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
- `"auth": "entra"` profiles (Azure AI Foundry) get a bearer token from optional
  `azure-identity` (`requirements-azure.txt`); keep that import lazy so the CLI stays
  standard-library. Token failures fail closed as a skipped model. `azureml/job.yml`
  runs the CLI in the Foundry workspace with the compute's managed identity; no keys.
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
