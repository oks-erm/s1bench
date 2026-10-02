# System 1 benchmark

A local dashboard and reusable CLI for Jev, Laya, Nimble, GPT and CLM. Compare classification/routing, NONE/clarification/escalation, binary checks and rubric scores. Plug models and labelled datasets into the GUI. No inference starts automatically.

## Start

1. [Download all files as ZIP](https://github.com/oks-erm/s1bench/archive/refs/heads/codex/system-one-benchmark.zip) and extract. Keep the complete folder.
2. Install **Python 3.10+** from [python.org](https://www.python.org/downloads/). On Windows, enable **Add Python to PATH**.
3. Open a terminal in the extracted folder:

| Device | Run |
| --- | --- |
| Windows | `py start.py` |
| macOS / Linux | `python3 start.py` |

First launch installs dashboard dependencies into a private `.venv`. Open **http://localhost:8501** if the browser does not open. The dashboard needs no GPU.

**Models & config:** enable available models, fill URLs/IDs/keys and save. **Plug your data:** upload JSONL or use the starter. **Run:** inspect the request budget, optionally run a one-case check, then Start. **Stop** prevents further requests; the in-flight request may finish or time out. Completed work is saved.

## Models

Use full inference URLs. An adapter describes a request/response shape; local models do not have to imitate OpenAI.

| Model | Adapter | Inference URL | Model ID / preparation |
| --- | --- | --- | --- |
| Jev | systemone | https://api.typesafe.ai/v1/systemone | jev-latest; GUI key or JEV_API_KEY |
| Laya | systemone | http://127.0.0.1:8000/v1/systemone | Blank permits server default; enter served ID if required |
| Nimble / Ollama | systemone | http://127.0.0.1:11434/v1/systemone | nimble; Ollama v0.35.0+; `ollama pull nimble` |
| GPT | openai | https://api.openai.com/v1/responses | Enter actual API model ID; GUI key or OPENAI_API_KEY |
| CLM v0.1 8B | systemone | http://127.0.0.1:8700/v1/systemone | clm-latest; serve checkpoint below |

**GPT Luna** is a display name, not an assumed API ID. Parameters start empty to avoid unsupported options causing HTTP 400. Responses effort, when supported: `{"reasoning":{"effort":"low"}}`. Chat effort: `{"reasoning_effort":"low"}`. For Chat use the full `/v1/chat/completions` URL. Generic Ollama chat uses the **ollama** adapter and `http://127.0.0.1:11434/api/chat`.

**CLM:** [checkpoint](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), [official serving guide](https://github.com/Contrastive-LM/CLM). On a suitable Linux/WSL2 GPU host install `pip install contrastive-lm`, then in separate terminals:

```sh
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --max-model-len 2048 --port 8090
clm-serve --host 127.0.0.1 --port 8700
```

Record actual checkpoint/encoder revisions, truncation, caches and GPU settings in provenance. The dashboard does not download weights. For remote servers use a reachable IP; localhost means the computer running the dashboard. Serving requirements depend on the model.

**Keys:** filled `api_key` is used directly; `api_key_env` is an environment variable's *name*. Keys stay in memory unless you check **Include entered keys when saving locally**. `config.json` is gitignored; run snapshots omit keys. Do not commit private configs/data.

Unavailable models skip independently. Full HTTP errors appear in **Cases → Warmup / connection checks** or **Failures → Full raw response**.

## Your data

Upload multiple JSONL files. One complete JSON object per nonempty line:

```json
{"id":"ticket-001","task":"intent","input":"Tell me a joke.","question":{"type":"choice","instructions":"Route support requests; NONE if no handler applies.","criteria":{"billing":"Charges or refunds","technical":"Bugs or outages","NONE":"Outside support"}},"expected":"NONE","cluster_id":"conversation-001","dataset":"support_holdout","dataset_role":"production_holdout","label_source":"human","review_status":"reviewed","critical":true}
```

Required: **id, task, input (or messages), question, expected**. Input accepts text, object or array. Question types:

- **choice:** criteria is an option map; expected is an option string.
- **noul:** yes/no question; expected is a boolean; optional true/false criteria.
- **score:** criteria is an ordered array of 2–10 levels; expected is a numeric zero-based level index.

New tasks need no code changes. Alternatively define `config.tasks[task].question` and omit per-case questions. Binary `threshold` defaults to 0.5; score success `tolerance` defaults to 0.5, both configured per task.

Group related variants/conversation turns/duplicates under the same **cluster_id**. Include reviewed NONE, near-matches, ambiguity, negation, quoted instructions, critical error directions and realistic histories. Optional globally unique `pair_id` plus `pair_relation` (equivalent, distractor, option_order, changed_fact) enables paired robustness metrics.

A copyable **AI data preparation prompt** is beside the shape in **Plug your data**. Attach your source file to your AI; expect one labelled JSONL file and a separate short summary. AI-inferred gold remains needs_review until reviewed.

CLI also supports a dataset manifest with relative paths:

```json
{"datasets":[{"name":"dev","path":"dev.jsonl","role":"development"},{"name":"holdout","path":"holdout.jsonl","role":"production_holdout"}]}
```

Results remain separate by dataset and use case.

## Trustworthy protocol

The starter creates **1,400 synthetic fixtures**: 1,200 standard cases across 12 tasks and 200 linked stress variants, approximately 120 independent template families. It checks plumbing/failure modes; it is not production evidence.

1. Freeze taxonomy, rubric, NONE/clarification/escalation policy, harmful error direction and business targets.
2. Review domain gold and separate development from an untouched production holdout. Start with 200–500 independent cases per priority use case; rare harmful errors need more. At 200 independent cases near 95% success, uncertainty is roughly ±3 percentage points. Zero harmful errors in 200 independent exposures still permits a one-sided 95% upper risk near 1.5%.
3. Send identical state/questions to each available deployment. Default: 2 warm-ups/model; whole-family assignment to 10 balanced batches; rotating model order. Score primary cases once and pool them. Ten batches are not ten independent datasets.
4. Repeat a stratified 100-case subset for 3 total observations: 200 extra calls/model. Measure consistency/critical flips separately; repeats do not inflate independent accuracy evidence. Explicit labelled pairs measure robustness.
5. Compare per-use-case success/F1, NONE/risk behavior, family-aware uncertainty and paired gap vs GPT, median/p95 latency, validity/errors and usage-based cost. Inspect critical failures; freeze IDs/effort. Confirm the chosen replacement on a fresh domain holdout before a controlled pilot.

All five profiles with starter defaults have an upper budget of **8,010 requests**. Reduce data/repetitions for shorter runs. Results includes a collapsible protocol and **Metric guide**.

Native SystemOne models receive typed questions; GPT/chat returns equivalent typed JSON decisions. This compares configured deployments, not long-form generation, real tool execution or full-agent task success.

Local **API** price is zero; infrastructure is separate. OpenAI rates use an exact community registry match with source/date. Unmatched Jev/custom prices need manual entry. Missing paid token usage stays unknown. Previous runs can be repriced with current GUI/config prices without inference. Cost coverage/currency/price basis are visible. Serial latency cannot establish fleet cost or throughput.

## Files and CLI

Launch **one file**, `start.py`. Keep `bench_ui.py`, `benchmark.py`, `starter_data.py`, `data_prompt.py`, `requirements.txt` and `config.example.json` beside it. First launch creates **one editable `config.json`**, `data/starter.jsonl`, and later `results/`.

Headless runner uses Python's standard library:

```sh
python benchmark.py --init
python benchmark.py --data data/starter.jsonl --config config.json --models jev,laya,nimble
python benchmark.py --data my_data.jsonl --config config.json
```

Windows: substitute `py`. macOS/Linux: `python3`. `--models` explicitly enables/selects aliases; otherwise enabled profiles are used. Optional `--limit 100`, `--out results`. There are no terminal key prompts.

Saved runs include raw responses/errors, frozen data/config, availability, summaries, batches, repeat/robustness results, confusion tables and unsigned SHA-256 fingerprints. Load a folder in Results; export CSV or standalone interactive HTML. Fingerprints detect accidental edits, not deliberate tampering or incorrect gold.

To access the dashboard from another device on your network: `python start.py --host 0.0.0.0`, then `http://YOUR-COMPUTER-IP:8501`. There is no built-in login; default localhost avoids exposure.

Maintainer checks use mock endpoints only:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```
