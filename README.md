# Dell Pro Max GB10 Demo Suite

A sales-demo benchmark suite for the **Dell Pro Max GB10** (Grace-Blackwell, aarch64,
sm_121). It benchmarks LLM / VLM / CNN / HPC workloads across precisions and surfaces
**business-relevant output** — bottleneck classification, TCO, and hardware-fit
recommendations — in a Streamlit UI organised **by failure-mode scenario**, not by
modality.

> **[`CLAUDE.md`](CLAUDE.md) is the authoritative spec.** This README is the front door;
> anything about *how* the engine or the guards behave lives there, and
> [`docs/CATALOG-2026.md`](docs/CATALOG-2026.md) is the append-only log of every model
> tried and why it was kept or dropped. Read that before re-litigating a model choice.

## Highlights

- **Phase-separated benchmarking** — load / prefill / decode timed independently, with
  warm-cache fairness so per-precision load times are comparable.
- **Precision sweep** — FP32 / FP16 / BF16 / FP8 / INT8 / FP4 / NVFP4. FP8 is real e4m3
  tensor-core compute via **torchao**; INT8 and FP4 are **bitsandbytes**; BF16/FP32 are
  native. A requested FP16 on a BF16-native checkpoint **runs as BF16** and says so,
  because a raw FP16 cast overflows BF16-trained activations and the run would still
  produce a plausible-looking number.
- **TCO engine** (`webui/helpers/tco_engine.py`) — Dell PowerEdge XE catalog, per-GPU
  spec sheet (`GPU_SPECS`), geometric KV-cache sizing, bandwidth-contention and
  fleet-coordination modelling, and a hardware-fit recommendation. Two rating profiles:
  - **LLM Inference** — per-user tok/s + $/MTok, anchored **per model** on measurements
    taken on this box (`webui/data/`), never one number reused across models.
  - **FinTech / Bandwidth-bound** — sizes a **Monte-Carlo** workload (resident paths ×
    bytes/path, timesteps, target paths/sec) to the fewest GPUs that fit the working set
    and meet the demanded memory bandwidth; ranks Dell platforms by $/B-paths.
- **On-device AI narrator** (`webui/helpers/on_device_ai.py`) — `Llama-3.2-3B-Instruct`
  narrates the results and the TCO recommendation *on the GB10*, no cloud.

## What actually runs (read this before adding a model)

The HF benchmark path runs **plain BF16** and **compressed-tensors
`nvfp4-pack-quantized`**. That is all. Measured on this box, not inferred from library
docs. Everything else — int4 `pack-quantized`, block-FP8, NVIDIA ModelOpt — *loads
successfully, reports success, and generates garbage*, which is the one result a
benchmark suite must never produce. `model_info.precision_compatible()` refuses them
before loading, keyed on the `(quant_method, format)` **pair**.

All of those checkpoints serve correctly under **vLLM**, which is how the sibling
projects on this box run them, and the gap is the interesting part of the demo rather
than a defect: `nvidia/Qwen3-8B-NVFP4` measures **38.8 tok/s** under vLLM against
**5.99** through the HF path, which decompresses to BF16 on every forward pass.

Two hazards worth knowing before you touch the catalog:

- **Adding a model means adding a `mem_guard._MODEL_PARAMS_B` entry.** That table is
  substring-matched with a 7.0 B fallback, and it has reported a 118 B checkpoint as 7 B
  three separate times. Over-committing this unified pool **deadlocks the kernel rather
  than raising a clean OOM** — it cost a hard power reset once.
- **A correct format label is not a correct checkpoint.** Acceptance for anything new is
  **load *and* generate**, with the output read. A successful load is not evidence.

The **vision track currently has no runnable VLM** — all six configured vision entries
are encoders (CLIP/ViT/DINO/ResNet/EfficientNet), and the two real VLMs on disk are
ModelOpt and refused. The image+prompt→text path is implemented and unreachable; see
`docs/CATALOG-2026.md` for the three attempts and why each failed.

## Layout

| Path | What |
|------|------|
| `setup.sh` | One-time host setup — verifies CUDA, installs cuDNN/TensorRT, builds the 4 venvs. Writes `.setup-state`. |
| `config/` | `model-lists.json` — the lists the UI *offers*. Distinct from `tco_engine.MODEL_CATALOG`, which also holds models too large to run here, for projection only. |
| `webui/` | The Streamlit app (`streamlit_app.py`) + `helpers/` (TCO, inference, narrator, bottleneck analyzer, mem guard, cloud pricing, measurement anchors). |
| `webui/data/` | **Measurement results, not config.** `vllm_reference.json` + `hf_anchors.json` are the only source of per-model `$/MTok`. |
| `webui/assets/` | `fixture-chart.png` + `.truth.json` — a chart with known values, so a vision model's answer can be checked rather than assumed. Generated; don't hand-edit. |
| `bin/` | Runnable checkers (`check_vlm.py`). |
| `scripts/` | `model-downloader.py`, `smoke_tco_capacity.py` (12 assertions), the measurement capture scripts, and `convert_nvfp4_to_ct.py`. |
| `docs/` | `CATALOG-2026.md`, the append-only decision log. |
| `workloads/` | Mostly empty — only `llm/simple_inference.py` exists. |

## Requirements

- Ubuntu 24.04 LTS **aarch64**, **Dell Pro Max GB10**, CUDA 13.0+ (several toolkits are
  installed; the venvs build against the `/usr/local/cuda` symlink)
- Python 3.12+, cuDNN 9.13, TensorRT 11.0
- 128 GB unified memory — CPU and GPU **share one pool**, ~100 GB usable. GPU memory *is*
  system RAM here, which is why the memory guards exist and why only one heavy workload
  runs at a time.

> **Not included in this repo:** model weights (`models/`, ~300 GB local and ~507 GB
> counting symlinked copies in sibling projects) and the Python venvs (`env/`, ~24 GB).
> Both are regenerated — see Quick start.

## Quick start

```bash
bash setup.sh                          # build the venvs (env/{hpc,llm,vlm,cnn})
python scripts/model-downloader.py     # pull weights per config/model-lists.json
                                       # (HF token at ~/.cache/huggingface/token or ~/hf-token)
python scripts/drop_model_page_cache.py   # after any large download — see note below
bash webui/launch.sh                   # launch the UI → http://localhost:8501
```

Each workload is venv-isolated:
```bash
source env/<workload>/bin/activate     # hpc | llm | vlm | cnn
```

Verify a change before trusting it:
```bash
env/llm/bin/python scripts/smoke_tco_capacity.py   # 12 assertions on the capacity model
env/llm/bin/python bin/check_vlm.py --no-gpu       # vision-track invariants
```

## Notes

- **Run `scripts/drop_model_page_cache.py` after a large download.** `mem_get_info()`
  reports MemFree, not MemAvailable, so page cache left behind by a download makes the
  next model load falsely refuse for lack of memory.
- `convert_nvfp4_to_ct.py` rewrites a ModelOpt NVFP4 export into compressed-tensors —
  this is why `nvidia/Qwen3-8B-NVFP4` runs at all. It renames quantization *suffixes*
  only, so it works for flat causal-LM architectures and **fails for nested multimodal
  ones**. Proven, not assumed.
- Keep new UI logic in `webui/helpers/`; `streamlit_app.py` orchestrates, helpers compute.
- HF downloads are resumable; the token is read at runtime (never committed).
- A figure shown in the UI always carries its provenance. If a model has no measured
  anchor the UI says so — it must never substitute another model's measurement.

## License

No license is granted. Internal demo material — confirm with the maintainer before
redistributing.
