# GB10 Demo Suite

Sales-demo benchmark app for NVIDIA GB10 / GH200 (aarch64) hardware. Benchmarks
LLM / VLM / CNN / HPC workloads across precisions (FP4, FP8, FP16, FP32, FP64)
and surfaces business-relevant output (bottleneck, TCO, hardware fit). The
Streamlit UI is organised by **failure-mode scenario**, not by modality.

> **Note:** The root `README.md` is stale (drifted on TensorRT version, model catalog, and UI organisation). Ignore it — this `CLAUDE.md` is the authoritative spec.

## System

- OS: Ubuntu 24.04 LTS aarch64 · GPU: NVIDIA GH200 / GB10 · CUDA 13.0 (`/usr/local/cuda-13.0`)
- Python 3.12+ · cuDNN 9.13 · TensorRT 11.0 (see `.setup-state` for installed versions)
- GB10 unified memory: 128 GB total, ~28 GB reserved → see `helpers/mem_guard.py` for usable budget.

## Layout

- `setup.sh` — one-time host setup (verifies CUDA, installs cuDNN/TensorRT, builds venvs). Writes `.setup-state`.
- `config/` — `model-lists.json` (hpc/llm/vlm/cnn model catalog) + per-workload `requirements-*.txt`.
- `env/{hpc,llm,vlm,cnn}` — isolated Python venvs, one per workload. **Not** under `venvs/`.
- `models/` — downloaded HF weights (large; gitignore-worthy). Models also live in `~/.cache/huggingface`.
- `scripts/model-downloader.py` — pulls models from HuggingFace per `config/model-lists.json`.
- `workloads/{hpc,llm,vlm,cnn}` — standalone workload scripts (most still stubs; `llm/simple_inference.py` exists).
- `logs/` — `setup.log`, `model-download.log`, `webui.log`, etc.
- `webui/` — the Streamlit app (primary surface).

## WebUI (`webui/`)

- `streamlit_app.py` — main app. Scenario-organised; imports everything from `helpers/`.
- `launch.sh` — entrypoint. Activates `env/llm`, ensures streamlit deps, runs on `http://localhost:8501`.
- `helpers/`
  - `llm_inference.py` — phase-separated LLM benchmarking (load / prefill / decode).
  - `vision_inference.py` — VLM/vision inference with load timing + phase tagging.
  - `hpc_compute.py` — HPC compute (matmul / bandwidth / reduction) with phase + bottleneck tags.
  - `memory_stress.py` — N-user concurrency + memory-spill detection.
  - `benchmark_utils.py` — `BenchmarkMetrics`, timing/memory profiling, `free_cuda_memory`.
  - `bottleneck_analyzer.py` — bottleneck classification via pynvml + observed metrics.
  - `mem_guard.py` — model footprint estimate vs. GB10 usable memory (`check_precisions`, `estimate_model_gb`).
  - `tco_engine.py` — Dell HW catalog + TCO calc + hardware-fit recommendation. `GPU_SPECS` is the per-GPU source of truth (memory, bandwidth, dense+sparse TFLOPS); each `DELL_SYSTEMS` entry names a `gpu_spec` and has VRAM/BW/TFLOPS derived from it (× `gpus_per_node`) by `_derive_system_specs()`. Don't hand-edit derived fields — change `GPU_SPECS`. **Flexible PCIe boxes** (XE7745 = 2–8× RTX PRO 4500, XE7740 = 2–8× RTX PRO 6000 BSE) set `flexible_gpus` + `min_gpus`/`max_gpus` + `chassis_price`/`gpu_price` + `chassis_tdp_w`/`gpu_tdp_w`; `calculate_tco` picks the fewest GPUs that serve the workload via `effective_node_gpus()` and prices/powers them via `node_price()`/`node_tdp_w()`. SXM/rack systems are always fully populated. **Bandwidth contention:** when multiple whole model copies co-reside on one physical GPU they share its memory bandwidth — `calculate_tco` scales per-user tok/s by `coresident**(_COPY_BW_SHARE_EXP-1)` (default exp 0.5 → aggregate ∝ √copies), so a big-VRAM/low-BW part (GB10 128 GB @ 273 GB/s) no longer gets "free" aggregate by packing copies. Reported as `TCOResult.bw_contention`; 1 copy/GPU or a copy that spans GPUs ⇒ no contention. **Fleet-coordination overhead:** CapEx gets a node-count-scaled surcharge (`min(_FLEET_COORD_CAP, _FLEET_COORD_PER_DOUBLING·log2(nodes))`, default +8%/doubling capped at +80%) on top of flat `add_infra_pct`, so a 200-box fleet doesn't cost-model like one node ×200 (networking/orchestration/spares/no cross-node batching). 1 node ⇒ 0%; reported as `TCOResult.fleet_overhead_pct`.
  - `business_metrics.py` — derive business output (e.g. cost) from raw metrics.
  - `model_info.py` — curated offline reference (architecture, params, designed precision) per model.
  - `on_device_ai.py` — local instruct model that narrates results ON the GB10; loads on demand, unloads after.

## Running

```bash
bash ~/gb10-demo/webui/launch.sh        # launch UI (uses env/llm venv) → localhost:8501
python scripts/model-downloader.py      # download/resume HF models per config
```

When working in a specific workload, activate its venv (`source env/<workload>/bin/activate`).

## Conventions

- Each workload is venv-isolated — don't assume one env has another's deps.
- HF token is cached at `~/.cache/huggingface/token`; large downloads are resumable.
- Precision support in LLM path is via bitsandbytes (FP4/FP8) + native FP16/BF16/FP32.
- Keep new UI logic in `helpers/` modules; `streamlit_app.py` orchestrates, helpers compute.
