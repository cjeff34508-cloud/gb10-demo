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
