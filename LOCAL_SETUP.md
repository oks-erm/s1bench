# Local models on Apple Silicon

Tested on `main` with an Apple M1 Pro, 16 GB unified memory, macOS 26.6.2 and
Python 3.11. The dashboard also runs on Windows/Linux; automatic management of
these three local runtimes is Mac-specific. Existing installations can skip to
[Start and switch models](#start-and-switch-models).

## Install on a fresh Apple Silicon Mac

Use a native ARM64 Python **3.11**, an internet connection and at least **40 GB of
free disk space** for environments, weights and download archives. The examples
below assume `python3.11` is installed and you are in the repository root. They
install the tested runtime versions without changing a system Ollama installation.
Downloads are large and may take time; complete them before starting a benchmark.

Create the dashboard environment and initialize the synthetic dataset:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python benchmark.py --init
```

Install the isolated inference environments and download the pinned checkpoints:

```sh
python3.11 -m venv .venv/local/laya
.venv/local/laya/bin/python -m pip install -r requirements-laya.txt
HF_HOME="$PWD/.venv/local/huggingface" HF_HUB_DISABLE_XET=1 \
  .venv/local/laya/bin/hf download convaiinnovations/laya \
  --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851

python3.11 -m venv .venv/local/clm
.venv/local/clm/bin/python -m pip install -r requirements-clm.txt
HF_HOME="$PWD/.venv/local/huggingface" HF_HUB_DISABLE_XET=1 \
  .venv/local/clm/bin/hf download mlx-community/CLM-v0.1-8B-MLX-8bit \
  --revision eb39b6c777c579720f7a1458f2614be4508a4e15 --local-dir .venv/local/clm-model
```

The CLM download includes its `clm_mlx` inference code. Our adapter imports that
pinned code directly; no separate CLM source installation is needed.

Install the project-local Ollama binary, verifying the official release checksum
before extracting:

```sh
mkdir -p .venv/local/ollama
curl --fail --location --retry 3 \
  https://github.com/ollama/ollama/releases/download/v0.35.0/ollama-darwin.tgz \
  --output .venv/local/ollama/ollama-darwin.tgz
echo '2608dbb0a0f0136a198db9d48b4f74ece55f452314a39452fca35b7cf20c2589  .venv/local/ollama/ollama-darwin.tgz' | shasum -a 256 -c - && \
  tar -xzf .venv/local/ollama/ollama-darwin.tgz -C .venv/local/ollama
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS="$PWD/.venv/local/ollama-models" \
  OLLAMA_NO_CLOUD=1 .venv/local/ollama/ollama serve
```

Leave that server running. In a second terminal, open the same repository folder:

```sh
OLLAMA_HOST=127.0.0.1:11435 .venv/local/ollama/ollama pull nimble
```

Wait for the pull to finish, then stop the server with Ctrl+C in its terminal.
`nimble` is a mutable upstream tag; the tested main weight blob is
`sha256:bbf1d6fc03bb0ed24d88f4c214ed7b5d1768aeb43d5cf433fb69eff0c8578013`.
Check `.venv/local/ollama-models/manifests/registry.ollama.ai/library/nimble/latest`
if you need to establish whether a later download uses the same weights.

Start the dashboard with `.venv/bin/python start.py`. In **Models & config**, use
**Load configuration** to upload `config.mac.example.json`, then save. This example
has the exact local endpoints, model IDs and checkpoint notes used here; all
participants start unchecked. Loading it replaces the dashboard's configuration,
so only do this for initial setup, or preserve existing custom settings first.
Set the hosted model IDs/keys for your account if using Jev or GPT. The example's
GPT ID is the one used in our comparison; your account must have access to it.

Select the models on **Run**, then use **Test selected models** first. It starts
and stops each installed local model and checks one response per selected model.
Select data and use **Run complete benchmark** for the shared report. Never leave
the standalone model server running during an automatic comparison.

## Start and switch models

For a comparison, select all desired models on **Run** and click **Run complete
benchmark**. The dashboard automatically starts and stops the installed local
profiles one at a time, and includes hosted models in the **same report**. Stop any
separately launched local server first. Jev and GPT need API keys in that dashboard
session or in its environment. Model settings, cases, rubrics, warm-ups and repeat
samples are frozen for the run. The report records model-by-model execution because
timing can be affected by the different execution times and provider conditions.

Results appear together on **Results**. Each results folder also contains
`report.html` and `summary.csv`, alongside raw responses and evidence. Missing or
failed models remain explicitly incomplete; they are never represented as successful.
Only one benchmark can run per results directory, including across browser tabs.
For recovery, `benchmark.combine_reports` can combine explicitly selected complete
models from saved runs with identical cases, scoring, batches and repeat samples.
It verifies evidence and preserves source references; combining makes no model calls.

For standalone use of just one model server:

On macOS, cloud storage can evict files inside this checkout, including hidden
model environments. If startup stalls and `ls -lO PATH` shows `dataless`, restore
the existing affected files before retrying. Finder's download/keep-downloaded
action or `brctl download PATH` requests restoration without changing package or
checkpoint versions. A folder request may leave children as placeholders; check
the actual runtime and tokenizer files. Confirm `/health` reports the intended
device and checkpoint revision before recovering missing benchmark requests.
Preserve the partial run and never repeat a model that already completed.

Double-click **Start Local.command**, then choose Laya, Nimble, or CLM.
Alternatively, from this folder:

```sh
.venv/bin/python local_models.py laya
# Or: .venv/bin/python local_models.py nimble
# Or: .venv/bin/python local_models.py clm
```

The launcher starts the dashboard if necessary at <http://127.0.0.1:8501>.
It starts the selected model server and preserves your saved settings.
Stop this standalone server before using the dashboard's automatic benchmark runner.
Use the standalone server for manual HTTP requests or the CLI without `--manage-local`.
The dashboard's **Test selected models** and **Run complete benchmark** manage
these installed profiles automatically, so stop the standalone server first.
Keep the terminal open; Ctrl+C stops that model. Stop an active benchmark before
switching models. To stop a separately running dashboard, use its own terminal.

Run **one model at a time**. In particular, Nimble and CLM should not both be loaded
on this 16 GB Mac. The launcher refuses to start when another local model endpoint
is listening or the pre-existing Ollama service reports a loaded model. It never
stops unrelated processes. A port conflict can also trigger this safeguard.

## Installed profiles

| Profile | Runtime | Local inference URL |
| --- | --- | --- |
| Laya English | Official `laya[serve]` 0.3.23, PyTorch MPS | `http://127.0.0.1:8000/v1/systemone` |
| Nimble | Project-local Ollama 0.35.0, Metal | `http://127.0.0.1:11435/v1/systemone` |
| CLM v0.1 8B | Approved community MLX 8-bit port | `http://127.0.0.1:8700/v1/systemone` |

The system's existing Ollama on port 11434 is unchanged. Hosted Jev and GPT profiles
are optional and require your API keys. No API key is required for the local profiles.
The local request timeout is 180 seconds to allow for initial loading on this Mac.

CLM model: [mlx-community/CLM-v0.1-8B-MLX-8bit](https://huggingface.co/mlx-community/CLM-v0.1-8B-MLX-8bit),
revision `eb39b6c777c579720f7a1458f2614be4508a4e15`.
This is an **unofficial quantized port**, not the authors' reference vLLM deployment;
answers and probabilities can differ. Its encoder is Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, 8-bit/group size 64, with the released
float32 projection heads. The local adapter uses first-2048-token truncation,
a 4096-entry projection cache, a 2048-token batching budget and a 256 MiB MLX
free-buffer cache limit. The latter prevents the default allocator cache from
retaining too much unused GPU memory during long runs on this Mac. Cached timings
must be interpreted accordingly. `local_clm_server.py` only adds HTTP transport;
the downloaded model package owns inference and answer scoring.

Laya currently uses the **English** checkpoint. Multilingual data needs a deliberate
checkpoint selection rather than assuming the English model supports it.
Its pinned revision is `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.
The official runtime warns that this checkpoint's `choice:11+` temperature is below
its supported range and clamps it to 0.5. Treat confidence on choices with 11 or more
options as uncalibrated; this warning is also recorded in the model profile.

## Files and storage

- `.venv/`: dashboard environment; `.venv/local/` contains separate model runtimes,
  weights, download logs, and runtime logs. These are all gitignored.
- `config.json`: local model profiles and benchmark settings, gitignored.
- `data/starter.jsonl`: generated synthetic screening dataset, created by the dashboard
  or `benchmark.py --init`.
- `results/`: saved benchmark runs, gitignored.

Do not delete `.venv` casually: it contains the downloaded model weights, not just
small Python dependencies. Restarting requires no model download once setup is complete.

## Verify

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile benchmark.py bench_ui.py start.py local_models.py local_runtime.py local_clm_server.py
zsh -n 'Start Local.command'
```

With the relevant standalone model running, check its full benchmark adapter:

```sh
.venv/bin/python benchmark.py --data data/starter.jsonl \
  --config config.json --models laya --connection-test --out results/local-smoke
```

Replace `laya` with `nimble` or `clm`. Inspect the saved `raw.jsonl` and availability
report as well as the exit status: a completed run may still skip an unavailable model.
This unscored request verifies integration; it does not establish model quality.
For automatic startup instead, add `--manage-local` and stop the standalone server.

To run all five automatically from Terminal (with hosted keys available):

```sh
.venv/bin/python benchmark.py --data data/starter.jsonl --config config.json \
  --models laya,nimble,clm,jev,gpt --manage-local
```

Verified on this Mac on 2026-10-02: all 38 regression tests passed, and each model
completed one warm-up plus three typed inference checks without API errors.

| Model | Valid responses | Correct primary cases | Saved report under `results/local-smoke/` |
| --- | --- | --- | --- |
| Laya | 4/4 | 3/3 | `20261002_103008_c5f432` |
| Nimble | 4/4 | 3/3 | `20261002_110350_dec070` |
| CLM MLX 8-bit | 4/4 | 1/3 | `20261002_110256_a7c9f2` |

All three saved reports passed the benchmark's evidence audit. These are tiny
integration checks, not a reliable ranking of the models. Their custom smoke-test
fixtures and reports are local artifacts, not files supplied by a fresh clone.

The subsequent complete comparison produced one audited report with **8,010 valid
responses**: 1,400 primary cases, 200 repeat calls and 2 warm-ups for each of the
five models. CLM was rerun with the bounded allocator cache, then combined with the
four complete models without repeating paid calls. Reports and weights remain local.

## Recovery

The model servers bind only to `127.0.0.1`. Keep them local: the CLM adapter is a small
single-user development server, not a public production service.

Hugging Face's Xet transfers stalled during initial setup; the local launcher uses
`HF_HUB_DISABLE_XET=1` for standard HTTP downloads. Download failures can be resumed:

```sh
HF_HUB_DISABLE_XET=1 .venv/local/clm/bin/hf download \
  mlx-community/CLM-v0.1-8B-MLX-8bit \
  --revision eb39b6c777c579720f7a1458f2614be4508a4e15 --local-dir .venv/local/clm-model
```

With the project Ollama server running:

```sh
OLLAMA_HOST=127.0.0.1:11435 .venv/local/ollama/ollama pull nimble
```

Laya resumes its cached download when started with the launcher.
Never include API keys, uploaded private data, or weight files in commits.

## API keys across sessions

In **Models & config**, leave the API key field blank and set the environment-variable
field to `OPENAI_API_KEY` or `JEV_API_KEY`. Keep **Include entered keys when saving
locally** unchecked. Save the configuration; it stores the variable name, not its
secret value. A filled API key field takes precedence over the variable.

For persistent secret storage on this Mac, use Keychain. Run only the commands for
providers you use. The final `-w` prompts securely, keeping the key out of shell history:

```sh
security add-generic-password -a "$USER" -s s1bench.OPENAI_API_KEY -w
security add-generic-password -a "$USER" -s s1bench.JEV_API_KEY -w
```

Before launching the dashboard from the same Terminal session:

```sh
export OPENAI_API_KEY="$(security find-generic-password -a "$USER" -s s1bench.OPENAI_API_KEY -w)"
export JEV_API_KEY="$(security find-generic-password -a "$USER" -s s1bench.JEV_API_KEY -w)"
.venv/bin/python start.py
```

The retrieval/export lines can also be added to a personal launcher before it starts
Python. The keys persist in Keychain; environment variables are inherited by newly
started child processes. Restart an already-running dashboard to receive new variables.
Creating a `.env` file alone has no effect: this project does not load it automatically.
The model downloads and local inference do not require these keys.
