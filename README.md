# System 1 benchmark

A local dashboard and CLI for comparing Jev, Laya, Nimble, GPT, CLM and other HTTP models on classification, yes/no decisions and rubric scores. Results show quality, latency, cost, errors and consistency by use case.

## Run

1. [Download the complete ZIP](https://github.com/oks-erm/s1bench/archive/refs/heads/codex/system-one-benchmark.zip) and extract it.
2. Install **Python 3.10+**. On Windows, enable **Add Python to PATH**.
3. Open a terminal in the extracted folder:

| Device | Command |
| --- | --- |
| Windows | `py start.py` |
| macOS / Linux | `python3 start.py` |

Open **http://localhost:8501**. First launch installs dependencies.

**This starts the dashboard only. Start local model servers separately.** In **Models & config**, add or enable models and fill their endpoint, model ID and API key. Save to the single local `config.json`. Models that are unavailable are skipped.

Included presets:

| Model | Endpoint |
| --- | --- |
| Jev | `https://api.typesafe.ai/v1/systemone` |
| Laya | `http://127.0.0.1:8000/v1/systemone` |
| Nimble / Ollama | `http://127.0.0.1:11434/v1/systemone` |
| GPT | `https://api.openai.com/v1/responses` |
| CLM | `http://127.0.0.1:8700/v1/systemone` |

Local serving: [Ollama](https://ollama.com/) · [CLM](https://github.com/Contrastive-LM/CLM). Use the exact model ID served by your endpoint.

Select data, then **Run → Start benchmark**. **Stop** prevents further requests; the current request may finish. Load saved runs in **Results** and export CSV or HTML.

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
