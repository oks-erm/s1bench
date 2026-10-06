# System 1 benchmark

A local dashboard and CLI for benchmarking System 1 models against GPT on classification, yes/no decisions and rubric scores. Results show quality, latency, cost, errors and consistency by use case.

## Run

1. [Download the complete ZIP](https://github.com/oks-erm/s1bench/archive/refs/heads/main.zip) and extract it.
2. Install **Python 3.10+**. On Windows, enable **Add Python to PATH**.
3. Open a terminal in the extracted folder:

| Device | Command |
| --- | --- |
| Windows | `py start.py` |
| macOS / Linux | `python3 start.py` |

Open **http://127.0.0.1:8501**. First launch creates `.venv` and installs dashboard
dependencies. Later launches reuse it. Keep the terminal open; Ctrl+C stops the dashboard.

For **Laya, Nimble and CLM on an Apple Silicon Mac**, follow the one-time
[local model setup](LOCAL_SETUP.md#install-on-a-fresh-apple-silicon-mac). Once installed
and configured, the dashboard starts and stops these models automatically, one at a
time, to fit a 16 GB Mac. Other local endpoints require separately running servers.
Model weights are downloaded separately; they are not included in this repository.

In **Models & config**, set endpoints, exact model IDs and API-key environment-variable
names, then save. Choose participants using the checkboxes on **Run**. All selected
models, including Jev and GPT, appear in **one comparison report**. Missing or failed
models are shown as incomplete. Configure keys before starting paid providers;
opening the dashboard does not run the benchmark.

Included presets:

| Model | Endpoint |
| --- | --- |
| Jev | `https://api.typesafe.ai/v1/systemone` |
| Laya | `http://127.0.0.1:8000/v1/systemone` |
| Nimble / Ollama | `http://127.0.0.1:11434/v1/systemone` |
| GPT | `https://api.openai.com/v1/responses` |
| CLM | `http://127.0.0.1:8700/v1/systemone` |

The Mac setup uses a separate Nimble server on **11435** and supplies a configuration
example with the matching model IDs. CLM on Mac uses the clearly labelled community
MLX 8-bit port, whose answers may differ from the reference model.

Local serving: [Ollama](https://ollama.com/) · [CLM](https://github.com/Contrastive-LM/CLM). Use the exact model ID served by your endpoint.

Select data, tick the models on **Run**, then click **Run complete benchmark**. It uses the entire dataset and configured consistency checks. **Test selected models** sends one short request to each ticked model. **Stop** prevents further requests; the current request may finish. Load saved runs in **Results** and export CSV or HTML.

## API keys and saved results

Use `JEV_API_KEY` and `OPENAI_API_KEY` in the environment of the terminal that launches
`start.py`; in Models & config, keep the literal key blank and set its variable name.
See [persistent macOS Keychain setup](LOCAL_SETUP.md#api-keys-across-sessions).
Keep **Include entered keys when saving locally** unchecked: keys typed into the
dashboard then stay session-only, and saving stores the variable name, not the key.
`.env` files are **not** loaded automatically. Restart the dashboard
after changing its environment.

Each run is saved under `results/` with `report.html`, `summary.csv`, raw responses,
frozen settings and evidence hashes. Reopen it from **Results → Recent run** after
restarting. Config, keys, downloaded weights, uploaded data and results stay local
and are excluded from Git. Back up results separately if you need to keep them.
**Download complete HTML report** creates one self-contained, offline report for
the entire run, regardless of the dataset or use case currently selected. Send that
file (or zip it for email); recipients open it in a browser with JavaScript enabled.
They do not need Python, Streamlit, model weights, API keys or internet access.

The viewer includes the business matrix, coloured quality/latency charts, cost and
forecast tables, fallback replay, confusion matrices, model profiles, run evidence,
protocol, metric guide, and all original cases and saved responses. It supports
dataset/use-case/reference selection, sortable tables, case search and pagination,
monthly volume scenarios and fallback-label checkboxes. CSV and complete data JSON
downloads are available inside the report. Choice and yes/no confusion matrices use
primary attempts only, with invalid/API errors separate; missing requests are not counted.

Prices, hosting rates and business targets are captured when exporting. Reference
comparisons use the same Python calculations as the dashboard, precomputed for each
model. Forecast volume and fallback replay run entirely in the browser. Setup, keys,
new inference and arbitrary repricing remain in the local app. The file includes
case inputs and responses, so choose a dataset appropriate for the intended recipients.
Regenerate an older export from its saved run; no model calls are needed. New runs
also save the complete viewer automatically as `report.html`.

Cost analysis uses complete current rates for the same model and endpoint, falling
back to the run's saved rates when current fields are blank. Explicit zero rates
remain valid overrides. For direct TypeSafe responses identifying `jev-1.13.0`,
the bundled [official list price](https://docs.typesafe.ai/models), verified
2026-10-02, is $0.042 per million input tokens and free output. It is not applied to
other returned versions or gateways. Manual rates take precedence. Missing usage
or unverified prices remain unknown. Results and exported HTML show the applied
rates, source and estimated total for all phases; these are not provider invoices.
Repricing preserves the original responses and frozen run evidence.

## With your data

Open **Plug your data** and upload labelled `.jsonl` files. The section includes a copyable AI preparation prompt and an example download. The included 1,400-case synthetic suite is for initial screening.

Each line must contain one case:

```json
{"id":"case-001","task":"intent","input":"Tell me a joke.","question":{"type":"choice","instructions":"Choose a support handler, or NONE if none applies.","criteria":{"billing":"Charges or refunds","technical":"Bugs or outages","NONE":"Outside support"}},"expected":"NONE","cluster_id":"conversation-001"}
```

Required fields: `id`, `task`, `input` (or `messages`), `question`, `expected`. Use `cluster_id` to group related cases. Choice gold is an option string; yes/no (`noul`) gold is a boolean; score gold is a zero-based rubric index. Use reviewed domain data for business decisions.

CLI alternative:

```sh
python benchmark.py --data your_data.jsonl --config config.json
```

Use `py` on Windows or `python3` on macOS/Linux. Keep the complete downloaded folder together.
For installed Mac models, add `--manage-local` to run them sequentially in one report:

```sh
.venv/bin/python benchmark.py --data data/starter.jsonl --config config.json \
  --models jev,laya,nimble,gpt,clm --manage-local
```

The dashboard creates the starter dataset on first launch. The CLI also supports
`python3 benchmark.py --init`. Select only models you have configured; hosted calls
may incur provider charges.

Requests and snapshots preserve JSON field and option order. Reports created before
the option-order fix used alphabetically sorted keys; their option-order stress
checks are excluded from current analysis when both saved orders are identical.
Old and new serialization policies cannot be mixed in one combined report.
Existing dataset files are retained. To create a fresh starter suite without
overwriting one, run this once with an unused filename, then select that file under
**Plug your data → Local path or dataset manifest**:

```sh
python3 -c 'from benchmark import jsonl_bytes; from starter_data import generate; open("data/starter_ordered.jsonl", "xb").write(jsonl_bytes(generate()))'
```

See [CLM investigation](CLM_DIAGNOSTIC.md) for the saved comparison's repeated CLM
answers, local checks, prompt sensitivity, and limits of the community MLX port.

## Development checks

```sh
python3 -m unittest discover -s tests -v
python3 -m py_compile benchmark.py bench_ui.py report_export.py report_server.py decision_analysis.py decision_ui.py shve.py starter_data.py data_prompt.py start.py local_models.py local_runtime.py local_clm_server.py
```

Use `.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows after
initial setup. Tests use fake model responses and do not make paid inference calls.
GitHub Actions runs the suite on Linux, Windows and macOS. See [AGENTS.md](AGENTS.md)
for architecture and maintenance notes.

## Protocol

Use a **paired evaluation design**: each model receives the same labelled cases and decision rubric. Freeze thresholds, model IDs, effort and business targets before evaluation. Report each dataset/use case separately; compare models only on complete, matched cohorts.

Defaults: **2 warm-up calls/model**, **10 balanced batches** with related cases kept together, and **100 stratified cases evaluated 3 times**. Primary outcomes are pooled once. Batches are partitions, not independent datasets; repetitions measure consistency without increasing the independent accuracy sample size. The short connection test is unscored.

**95% confidence intervals** use Wilson intervals for independent cases or a **cluster bootstrap** over case families. Differences from the reference model use paired inference. Grouping avoids pseudoreplication; insufficient or degenerate evidence remains inconclusive. Frozen data/config snapshots and a fixed sampling seed support reproducibility.

| Measure | Metrics |
| --- | --- |
| Classification | Task success (`success_rate`), macro F1, NONE precision/recall |
| Numeric decisions | Brier score for binary probabilities; mean absolute error (MAE) for rubric scores |
| Risk and reliability | Unsafe-decision rate per risk exposure, critical failures, valid-answer rate, API errors |
| Stability | Repeat agreement, all-repetitions-correct rate, critical repeat flips; labelled robustness pair success |
| Latency | Client p50/p95 after warm-up; attempt p95 includes failed requests |
| Economics | API cost/1,000 decisions, cost/correct decision, cost coverage; separate hosting scenarios |

API failures and invalid answers count against task success. Read Brier/MAE alongside validity coverage. Costs use recorded token usage; missing paid usage remains unknown. Use a **reviewed domain holdout** separate from development data: synthetic fixtures establish screening results, not production performance. Intervals are unadjusted for multiple comparisons; confirm model selection on a fresh holdout.

## SHVE decision workflows

See [SHVE_GUIDE.md](SHVE_GUIDE.md) for package validation, the four-model protocol,
input-only deterministic baselines, directional risk scoring, acceptance thresholds,
workflow cost assumptions and offline report regeneration. New **Acceptance and risk**,
**Workflow economics**, **Dataset diagnostics** and **Deterministic baselines** report
sections use saved responses without additional inference. Historical label-based
coverage, frozen evidence and existing report formats remain supported.
