# Model catalog refresh — 2026-11-05

## Why

The suite benchmarked 2024-era **dense** models — TinyLlama-1.1B, Llama-3.2-3B,
Qwen2.5-7B/14B/32B, Mistral-7B-v0.1, Phi-4, Mixtral-8x7B — with one modern
entry, `nvidia/Qwen3-8B-NVFP4`. Meanwhile every other project on this box
serves 2026 **MoE at FP8/FP4**.

Dense BF16 is the worst case for 128 GB at 273 GB/s. A catalog of only dense
models showcased the hardware's weakest axis while the box's own sibling
projects demonstrated its strongest.

## Added

Four cost nothing to add: sibling projects already hold the weights, and
`models/llm-models/` now carries symlinks to their snapshots.

| Catalog name | Source | On-box evidence |
|---|---|---|
| `Qwen3.6-35B-A3B` | symlink -> gb10-markets | 61 tok/s single-stream with MTP (53.8 without), 84% draft acceptance, peak 56.2 GiB at 4-way, 10.0 KiB/token KV |
| `Nemotron-3.5-Lightning-30B-A3B` | symlink -> gb10-nemotron | NVIDIA positions it *for DGX Spark*; 3.0 KiB/token KV, 1M context |
| `Laguna-S-2.1` | symlink -> gb10-laguna | 92.9 GiB measured on disk, **not** the published ~71 GB; 18 tok/s |
| `Qwen3.8-27B` | symlink -> HF cache | dense-at-FP8 control, so MoE-vs-dense is like-for-like on precision |
| `Gemma-4-26B-A4B` | downloaded, 18.8 GB | small-MoE control; community report gives 52 tok/s on GB10 |
| `Nemotron-3-Super-120B-A12B` | downloaded, 80.4 GB | NVIDIA publishes a DGX Spark deployment guide; was already projected in the TCO catalog with no way to measure it |

Every architecture figure in `MODEL_ARCH` was read from the checkpoint's own
`config.json`, not from published specs. That mattered: all four of the newest
models are hybrids whose KV lives on a minority of layers, and all four have a
`head_dim` that is **not** `hidden / q_heads` (Laguna is 3072/48 = 64 derived
vs 128 real). See `docs/KV-MODEL-CHANGE.md`.

## Retired from the benchmarkable list

| Model | Weights on disk | Why |
|---|---|---|
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 87 GB | 2023 MoE; superseded as the MoE exemplar by any A3B |
| `Qwen/Qwen2.5-32B-Instruct` | 62 GB | dense 32B BF16 is the anti-showcase; also duplicated in the HF cache |
| `mistralai/Mistral-7B-v0.1` | 28 GB | stored at 28 GB for a 7B model, i.e. FP32, and it is a *base* model, not instruct |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2.1 GB | toy; the ladder's bottom rung is Llama-3.2-3B |

Plus two unreferenced narrator-dir checkpoints: `Qwen--Qwen3.5-35B-A3B` (67 GB
BF16) and `Sehyo--Qwen3.5-35B-A3B-NVFP4` (24 GB), both superseded by the
Qwen3.6-35B-A3B-FP8 symlink.

**Removal from the list is not removal from `MODEL_CATALOG`.** `TinyLlama-1.1B`
and `Mixtral-8x7B` are named in `scripts/smoke_tco_capacity.py` and in all 90
golden-baseline rows; their catalog entries stay. `MODEL_CATALOG` is explicitly
inclusive of models too large to run here, so projection-only entries are
normal. Only `config/model-lists.json` and the UI selector changed.

## Kept deliberately

`nvidia/Qwen3-8B-NVFP4` (the only existing FP4 proof — 36.4 vs 13.1 tok/s is
the suite's cleanest result), `unsloth/Llama-3.2-3B-Instruct` (the on-device
narrator, `on_device_ai.py:25` — removing it silently disables narration),
`Phi-4`, `Qwen2.5-7B-Instruct` (the measured vLLM reference anchor) and
`Qwen2.5-14B-Instruct` (the dense ladder).

## The UI selector is now presence-filtered

`config/model-lists.json` is the single source of truth, and the selector
offers only entries that `_local_path()` actually resolves on disk, naming any
that are missing in a caption. Previously the list was hardcoded in
`streamlit_app.py`; a retired or half-downloaded entry would have made
`from_pretrained()` fall back to the HF id and pull tens of GB mid-demo.

## A safety bug found on the way in

`mem_guard.params_b()` matches lowercase substrings and falls back to 7.0B.
Four of the six additions defeated it — `Laguna-S-2.1` and
`Nemotron-3-Super-120B` matched nothing, and `Qwen3.8-27B` matched `"7b"`
inside `"27b"`. `fits_in_memory()` therefore put Laguna at 15.4 GB BF16 and
returned **fits=True** for a 92.9 GiB checkpoint, so the UI would have
attempted the load and over-committed the unified pool — the same failure that
hard-hung this box on 2026-06-30. Fixed with explicit entries ahead of the
generic suffixes; Laguna now estimates 260.3 GB and is refused.

---

# Cloud pricing refresh — 2026-11-05

## What was wrong

Every Claude price in `cloud_pricing.py` was **correct for the model it named** —
Opus 4.8 $5/$25, Sonnet 4.6 $3/$15, Haiku 4.5 $1/$5 all verified. The problem was
that the models were a generation behind, and the current generation is *cheaper*:

| Was quoted | $/1M | Current equivalent | $/1M | Delta |
|---|---|---|---|---|
| Claude Opus 4.8 | 5.00 / 25.00 | Claude Opus 5.5 | **4.00 / 20.00** | 20% cheaper |
| Claude Sonnet 4.6 | 3.00 / 15.00 | Claude Sonnet 5.5 | **2.00 / 10.00** | 33% cheaper |
| Claude Haiku 4.5 | 1.00 / 5.00 | Claude Haiku 5.5 | **0.10 / 0.50** | **10x cheaper** |
| — | — | Claude Fable 5.1 | 10.00 / 50.00 | no row existed |

So the table overstated hosted-API cost — which flattered on-prem by accident.
That is the same failure mode as an unlabelled projection: a number that leans
one way without saying so.

## The stance, stated

The comparison is now **deliberately conservative toward on-prem, and says so on
screen** (`STANCE_LABEL`, rendered as a warning banner above the comparison).
Four levers, all of them real published pricing structure:

1. **Long-context tiers applied.** Claude Haiku 5.5 is 5x above a 100K prompt;
   Gemini 3.1 Pro is ~2x above 200K. The TCO tab **defaults to 131,072 context**
   and goes to 524,288, so the higher tier is the normal case here, not an edge
   case. The previous module ignored tiers entirely and therefore understated
   cloud cost at every default setting — this lever makes it *more accurate* and
   higher at the same time.
2. **Thinking tokens billed as output.** Adaptive thinking is on by default on
   every current model, so a reasoning workload is billed for more output than it
   returns. `THINKING_OUTPUT_MULTIPLIER = 1.5` is the **only estimated input in
   the module** and is labelled as an assumption, not a measurement; set it to
   1.0 for a thinking-free workload.
3. **Frontier-class price point.** `CLOUD_ANCHORS` moved from
   `point = Gemini 3.5 Flash` / `high = Claude Sonnet 4.6` (Sonnet is mid-tier,
   mislabelled "frontier") to `point = Claude Opus 5.5` / `high = Claude Fable 5.1`,
   with `low = Claude Haiku 5.5` keeping the honest floor visible.
4. **Cache reads and the 50% Batch discount are NOT applied.** Both are real and
   both would lower cloud cost. Omitting them is conservative and is recorded so
   the omission is visible rather than accidental.

Effect at the default 131,072 ctx / 16,384 out: the quoted price point goes from
**$17.00** to **$41.33 per 1M output tokens**, with a **$103.33** ceiling — every
figure a published list price.

## What was NOT done, deliberately

**The stale Haiku 4.5 rate was not kept just because it was higher.** At its real
>100K tier Haiku 5.5 costs $5.17/1M out versus the stale Haiku 4.5's $10.33 — so
correcting it *lowered* the floor. Using a wrong number because it favours the
conclusion is fabrication, and it is the first thing a customer's procurement
team would catch. The stance comes from pricing structure, never from stale data.

**Gemini rows were not changed.** Third-party trackers broadly agree with the
rates already in the file (3.1 Pro $2/$12, 3.5 Flash $1.50/$9), but Google's own
pricing page could not be reached on 2026-11-05, so every Gemini row now carries
`verified: False` and the UI says the Gemini rates are unverified. One live risk
is recorded: Gemini 3.8 Flash is reported to be on an introductory rate ending
**2027-01-01**, which would silently invalidate the comparison on New Year's Day.

## Also verified, not changed

The box's own Claude spending is correctly priced. `second_opinion.py:23` carries
`PRICE = {"input": 5.00, "output": 25.00}` for `claude-opus-5`, which matches the
first-party rate; the four logged second opinions ($0.065-$0.09) are accurate.
Moving that path to `claude-opus-5-5` would cut it ~20% and require the constant
to become 4.00/20.00.

---

# C0 downloads complete — 2026-11-05 12:03

All four exact manifest matches, `rc=0`, 1 h 36 m total:

| Repo | On disk | Expected |
|---|---|---|
| `nvidia/Gemma-4-26B-A4B-NVFP4` | 18.8 GB | 18.8 GB |
| `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` | 24.1 GB | 24.1 GB |
| `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` | 75.1 GB | 75.1 GB |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 80.4 GB | 80.4 GB |

## Nemotron-3-Super is 8 attention layers of 88

Its config carries no `layer_types`, but it does carry
`hybrid_override_pattern` — the NemotronH convention where `*` is attention,
`M` is mamba and `E` is MLP/MoE:

```
MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME
```

**8 attention, 40 mamba, 40 MoE.** Billing all 88 layers would have over-stated
its KV by **11x**. At fp8 it is **4.0 KiB/token** — a 120B model with less KV per
token than a 3B dense one, because 80 of its 88 layers hold no growing cache.

With that row added, **no LLM catalog entry is on the legacy KV fallback.**

## Measured checkpoint size now beats the flat estimate

A flat bytes-per-param figure is wrong for mixed-precision NVFP4 releases, which
commonly quantize only the expert projections and leave the rest BF16:

| Model | flat 0.5 B/param | measured on disk | ratio |
|---|---|---|---|
| Laguna-S-2.1 | 59.0 GB | **92.9 GiB** | 1.6x |
| Nemotron-3-Super-120B | 60.0 GB | **80.4 GB** | 1.3x |

`MODEL_CATALOG` entries now take an optional `disk_gb` per precision, and
`_weights_and_raw()` prefers it. Under-stating weights on a 121 GB unified pool
is the dangerous direction — and the corrected figures match observed reality:

- **Laguna**: 108.1 GB weights + 6.4 GB KV at 256K = **114.5 GB against 100 GB
  usable → will not fit.** That matches the live box, where Laguna needs
  `--gpu-memory-utilization 0.92` and currently fails to start.
- **Nemotron-3-Super**: 87.1 + 1.1 = **88.1 GB → fits, but tightly**, matching
  the "monitor memory closely" caution on a 120B NVFP4 on this hardware.

## model_info.py gap closed

The C2 commit added six models to the catalog without rows in
`model_info.py`'s `MODEL_INFO` / `DESIGNED_PRECISION`, so the UI's detail panel
and `precision_compatible()` had nothing for them. All six now have entries
carrying the measured footprint, the real attention-layer split, and the
per-model footguns (triton MoE backend for Qwen3.6; Bearer-only auth and
explicit `enable_thinking` for Laguna). **0 catalog entries missing either
table.**

Final state: LLM 11/11 on disk and offered, VLM 6/6 on disk, harness 12/12,
app serves HTTP 200.
