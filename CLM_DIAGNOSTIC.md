# CLM investigation — 2 October 2026

The saved five-model comparison shows **answer concentration**, not a first-option
selection bug. This investigation concerns the approved community MLX 8-bit port,
not an independently tested full-precision CLM deployment.

## Saved comparison

Run `20261002_135147_combined_c95664` contains 1,400 primary cases per model.
CLM answered 562 correctly (40.14%). Among its 583 choice cases, it selected the
first transmitted option only 6 times (1.03%). The original requests sorted JSON
keys, so transmitted order is recoverable from the frozen snapshots.

Repeated predictions explain the unusual confusion matrices:

| Task | Most frequent prediction | Cases |
|---|---|---:|
| Intent | sales | 104 / 118 |
| Referent | NONE | 107 / 118 |
| Reply gate | NONE | 100 / 112 |
| Retrieval | B | 104 / 118 |
| Action | LOOKUP_ORDER | 84 / 117 |

## Checks performed locally

- Replayed one incorrect and one correct standard case from each of the five
  choice tasks through the same HTTP adapter. All 10 matched the saved answer and
  every saved probability exactly.
- Reversed option order for those 10 requests. All choices stayed unchanged;
  the largest probability difference was floating-point rounding (< 3e-16).
- Compared the downloaded schema with the [pinned official source](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/src/clm/schema.py).
  Executable schema code is identical; only its opening documentation differs.
  The adapter passes the state and question through, without expected labels.
- Counted untruncated tokenizer lengths for all 1,400 cases: the longest state
  plus instructions was 151 tokens; the longest candidate was 11 tokens.
  None reached the 2,048-token limit.
- Checked the official department example and a simple tides question. CLM chose
  billing and the Moon's gravitational pull, respectively.
- Tested three messages with both object and string states and with original
  versus shortened instructions, keeping candidate descriptions fixed. For
  `My invoice was charged twice.`, shortening the instruction to
  `Choose the primary support intent.` changed sales to billing with either state
  representation. For the outage example, both shorter instructions and a string
  state were needed to obtain technical in this small controlled sample.
- Cleared the projection cache and compared batched, single-text and reversed
  candidate embeddings on an outage case. All selected sales. Single-text
  embedding changed probabilities by at most 0.01086; cold reversal by < 2e-16.
  This is a narrow batching check, not full numerical parity certification.

These 43 local diagnostic answers support sensitivity to wording/representation,
rather than a lost input, tie-to-first, or stale-response explanation. They do not
establish that one prompt change improves the entire benchmark. No full-precision
GPU reference was run, so quantization or other port-specific effects remain
unresolved. The diagnostic requests are separate from benchmark scores, and no
hosted model calls were made during the investigation.

## Separate benchmark defect fixed

Earlier versions used sorted-key JSON for requests and dataset exports. This
silently undid the starter dataset's option-order perturbations for every model.
New requests, chat payloads and snapshots preserve supplied order. The manifest
records the serialization policy, and combining reports rejects different policies
or different ordered inputs. Legacy reports can still be combined with each other.

Analysis now excludes labelled option-order pairs whose saved criteria have the
same order, and displays a warning in the dashboard and HTML exports. The original
comparison has 20 such pairs. Primary task accuracy remains an observation of the
original shared inputs; its option-order robustness figures are not valid evidence
of order invariance. Historical raw responses and frozen snapshots are retained.

Do not silently shorten prompts only for CLM and substitute its score into the
existing comparison. A prompt experiment needs a separately labelled run and a
predeclared input policy, with all relevant models compared on matched inputs.
