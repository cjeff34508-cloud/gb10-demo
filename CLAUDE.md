# Dell Pro Max GB10 Demo Suite

Sales-demo benchmark app for Dell Pro Max GB10 (aarch64) hardware. Benchmarks
LLM / VLM / CNN / HPC workloads across precisions (FP4, FP8, FP16, FP32, FP64)
and surfaces business-relevant output (bottleneck, TCO, hardware fit). The
Streamlit UI is organised by **failure-mode scenario**, not by modality.

> **Note:** This `CLAUDE.md` is the authoritative spec. The root `README.md` was rewritten
> 2026-11-06 and is now accurate, but it is a front door — it summarises and defers here for
> anything about how the engine or the guards actually behave.

## System

- OS: Ubuntu 24.04 LTS aarch64 · GPU: Dell Pro Max GB10 (Blackwell **sm_121**) · CUDA 13.0+
  (several toolkits are installed — `cuda-13.0/.3/.4/.5` — so don't assume `CUDA_HOME`; the venvs
  were built against the `/usr/local/cuda` symlink)
- Python 3.12+ · cuDNN 9.13 · TensorRT 11.0 (see `.setup-state` for installed versions)
- Dell Pro Max GB10 unified memory: 128 GB total, ~28 GB reserved → ~100 GB usable; see
  `helpers/mem_guard.py`. **GPU memory IS system RAM here.**
- ⚠️ **Adding a model means adding a `mem_guard._MODEL_PARAMS_B` entry.** That table is
  substring-matched and its fallback is 7.0 B, which has reported a 118 B checkpoint as 7 B
  three separate times (`laguna`→nothing, `qwen3.8-27b`→"7b", `qwen3-0.6b`→nothing). An
  under-stated footprint passes the fits-in-memory check, and over-committing this unified pool
  **deadlocks the kernel rather than raising a clean OOM** — it took a hard power reset on
  2026-06-30. Mixed-precision NVFP4 releases also need a `_MEASURED_GB` row, because only part
  of the checkpoint is 4-bit so `params × 0.5` comes in *below* the real file.

## Layout

- `setup.sh` — one-time host setup (verifies CUDA, installs cuDNN/TensorRT, builds venvs). Writes `.setup-state`.
- `config/` — `model-lists.json` (the `hpc`/`llm`/`vlm`/`cnn` lists the UI *offers*; `hpc` is
  empty) + per-workload `requirements-*.txt`. Distinct from `tco_engine.MODEL_CATALOG`, which
  deliberately also holds models too large to run here, for projection only.
- `env/{hpc,llm,vlm,cnn}` — isolated Python venvs, one per workload. **Not** under `venvs/`.
- `models/` — downloaded HF weights (large; gitignore-worthy). Models also live in `~/.cache/huggingface`.
- `scripts/`
  - `model-downloader.py` — pulls models from HuggingFace per `config/model-lists.json`.
  - `smoke_tco_capacity.py` — **12** standalone assertions on the TCO capacity model.
  - `measure_hf_anchors.py` → `webui/data/hf_anchors.json` (single-stream tok/s, HF path).
  - `capture_vllm_reference.py` / `recapture_reference.sh` → `webui/data/vllm_reference.json`
    (concurrency sweep against a live vLLM container; one model at a time, serialised).
  - `recalibrate_batch_exp.py` — least-squares fit of `_BATCH_BW_SHARE_EXP` to the captures.
  - `drop_model_page_cache.py` — `posix_fadvise(DONTNEED)` on weight files. **Run after any
    large download**: `mem_get_info()` reports MemFree, not MemAvailable, so page cache from a
    download makes the next load falsely refuse for lack of memory.
  - `make_chart_fixture.py` — draws `webui/assets/fixture-chart.png` + truth from the captures.
  - `convert_nvfp4_to_ct.py` — rewrites a **ModelOpt** NVFP4 export into compressed-tensors
    `nvfp4-pack-quantized`. This is why `nvidia/Qwen3-8B-NVFP4` runs at all. It renames
    quantization *suffixes* only, so it works for flat causal-LM archs and **fails for nested
    multimodal ones** (proven on Gemma-4: loads, then generates nothing). See CATALOG-2026.md.
  - `periodic_state_saver.py` — **orphaned**; its docstring promises a cron job that does not
    exist (`crontab -l` is empty). Wire it up or delete it.
- `workloads/{hpc,llm,vlm,cnn}` — three of the four are **empty**; the only file is
  `llm/simple_inference.py`.
- `bin/` — runnable checkers. `check_vlm.py` (7 assertions; `--no-gpu` skips 5–7).
- `docs/` — `CATALOG-2026.md` is the **append-only decision log**: every model tried, what it
  did, and why it was kept or dropped. Read it before re-litigating a model choice.
- `webui/data/` — **measurement results, not config.** `vllm_reference.json` (vLLM concurrency
  sweeps) and `hf_anchors.json` (single-stream HF tok/s). These are the only source of per-model
  `$/MTok`; losing them means every model's cost column collapses to one shared number.
- `webui/assets/` — `fixture-chart.png` + `.truth.json`, a chart with known values so a vision
  model's answer can be checked. **Generated** by `scripts/make_chart_fixture.py` from
  `vllm_reference.json` — never hand-edit the values.
- `backups/`, `state_snapshots/` — local scratch, gitignored.
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
  - `mem_guard.py` — model footprint estimate vs. Dell Pro Max GB10 usable memory (`check_precisions`, `estimate_model_gb`).
  - `tco_engine.py` — Dell HW catalog + TCO calc + hardware-fit recommendation. `GPU_SPECS` is the per-GPU source of truth (memory, bandwidth, dense+sparse TFLOPS); each `DELL_SYSTEMS` entry names a `gpu_spec` and has VRAM/BW/TFLOPS derived from it (× `gpus_per_node`) by `_derive_system_specs()`. Don't hand-edit derived fields — change `GPU_SPECS`. **Flexible PCIe boxes** (XE7745 = 2–8× RTX PRO 4500, XE7740 = 2–8× RTX PRO 6000 BSE) set `flexible_gpus` + `min_gpus`/`max_gpus` + `chassis_price`/`gpu_price` + `chassis_tdp_w`/`gpu_tdp_w`; `calculate_tco` picks the fewest GPUs that serve the workload via `effective_node_gpus()` and prices/powers them via `node_price()`/`node_tdp_w()`. SXM/rack systems are always fully populated. **Capacity model (`calculate_tco(capacity_mode=…)`, default `"shared"`):** how concurrent sessions pack into VRAM. `"shared"` = continuous batching / paged attention — **one** weight replica per GPU (or per `gpus_per_model`-GPU tensor-parallel group), and the remaining VRAM is packed with per-session KV: `sessions/group = floor((group_vram − weights_once) / kv_per_session)`. `"replicated"` = the naive worst case — every session carries its own full weight copy (`floor(vram / (weights+kv))`, 1 session/copy). `session_footprint()` splits `(weights_once, kv_per_session)` from the **same** terms as `model_memory_gb` (which is untouched = their sum). The UI radio flips the whole sizing pipeline (sessions/node → nodes → CapEx → TCO → $/MTok → ratings). Shared is the realistic default (≈30× the aggregate concurrency of replicated on GB10, matching `webui/data/vllm_reference.json`). **Bandwidth contention:** sessions/copies co-resident on one physical GPU share its memory bandwidth — `calculate_tco` scales per-user tok/s by `coresident**(share_exp-1)`, where `share_exp = _COPY_BW_SHARE_EXP` (0.5 → aggregate ∝ √copies) in replicated mode and `_BATCH_BW_SHARE_EXP` (**0.90**, near-linear — batching amortises the weight reads) in shared mode. The 0.90 exponent is **calibrated + pinned** by `scripts/smoke_tco_capacity.py` assertion 5 (analytic shared aggregate within [0.5, 2.0]× the real GB10 vLLM captures). A big-VRAM/low-BW part (Dell Pro Max GB10 128 GB @ 273 GB/s) no longer gets "free" aggregate by packing sessions. Reported as `TCOResult.bw_contention`; 1 session/GPU or a copy that spans GPUs ⇒ no contention. **Smoke tests:** `python scripts/smoke_tco_capacity.py` (**12** assertions: 1–7 the original set incl. golden-baseline back-compat for replicated mode via `scripts/tco_golden_baseline.json`, 8–12 the KV-geometry invariants — expert-count independence, linearity in context, MLA, total-param capacity, and reproduction of both on-box KV measurements, 48.0 and 10.0 KiB/token); regenerate the baseline only with `--generate` after an intended replicated-path change. **Fleet-coordination overhead:** CapEx gets a node-count-scaled surcharge (`min(_FLEET_COORD_CAP, _FLEET_COORD_PER_DOUBLING·log2(nodes))`, default +8%/doubling capped at +80%) on top of flat `add_infra_pct`, so a 200-box fleet doesn't cost-model like one node ×200 (networking/orchestration/spares/no cross-node batching). 1 node ⇒ 0%; reported as `TCOResult.fleet_overhead_pct`.
  - `business_metrics.py` — derive business output (e.g. cost) from raw metrics.
  - `cloud_pricing.py` — hosted-API rates for the on-prem-vs-cloud panel. `CLOUD_ANCHORS`
    (low/point/high), `effective_rates()`, `THINKING_OUTPUT_MULTIPLIER` (the only estimate in
    the file — everything else carries a `verified` flag) and `STANCE_LABEL`, which is rendered
    as a banner because the pricing stance is deliberately **high-side / on-prem-favouring**.
  - `vllm_reference.py` — reads `webui/data/*.json`. `best_anchor(catalog_name)` is the single
    accessor for a per-model throughput anchor and **prefers vLLM over HF**, always returning a
    provenance string so a figure is never shown without saying which stack produced it.
    `capture_for_model()` maps an HF id or catalog key to its capture, so the serving panels
    follow the model actually benchmarked. No anchor ⇒ the UI says so; it must never
    substitute another model's measurement.
  - `model_info.py` — curated offline reference (architecture, params, designed precision) per model.
  - `on_device_ai.py` — local instruct model that narrates results ON the Dell Pro Max GB10; loads on demand, unloads after.

## Running

```bash
bash ~/gb10-demo/webui/launch.sh        # launch UI (uses env/llm venv) → localhost:8501
python scripts/model-downloader.py      # download/resume HF models per config
```

When working in a specific workload, activate its venv (`source env/<workload>/bin/activate`).

## Conventions

- Each workload is venv-isolated — don't assume one env has another's deps.
- HF token is cached at `~/.cache/huggingface/token`; large downloads are resumable.
- Precision support in LLM path: **FP8** = real e4m3 tensor-core compute via **torchao** (`TorchAoConfig` + `Float8DynamicActivationFloat8WeightConfig`; validated on this Dell Pro Max GB10 sm_121 aarch64 — `Float8Tensor`/`float8_e4m3fn` weights, ~1 byte/param, coherent output; note it trades memory for speed in eager mode, i.e. smaller but not faster than BF16 on this box). INT8/FP4 via bitsandbytes; native BF16/FP32. **FP16 auto-runs as BF16 for BF16-native models** (every catalog LLM is BF16-native; a raw FP16 cast overflows BF16-trained activations → NaN/garbage), labeled `FP16 → BF16` in results; true FP16 is kept only for FP16-native checkpoints (CLIP/ViT vision path). (bitsandbytes has **no** FP8 mode — the old "FP8 via bitsandbytes" note was wrong; FP8 is torchao.)
- Keep new UI logic in `helpers/` modules; `streamlit_app.py` orchestrates, helpers compute.
- **FP16→BF16 substitution only fires if the model has a `model_info.DESIGNED_PRECISION` entry.**
  `llm_inference` asks `designed_precision()` whether the checkpoint is BF16-native; no entry means
  no substitution and a raw `float16` cast, which overflows BF16-trained activations. The run still
  times, so the result looks plausible and is meaningless. Adding a BF16 model = adding that entry.

## Pre-quantized checkpoints: what actually runs (measured, not advertised)

**The HF benchmark path runs exactly two things: plain BF16, and compressed-tensors
`nvfp4-pack-quantized`. Nothing else.** This is measured on this box (sm_121, aarch64,
transformers 5.5.4), and it is not what the library advertises. The failure mode is the dangerous
one: transformers prints *"Unknown quantization type … we will skip the quantization"*, reinterprets
packed weights as BF16, **reports a successful load**, and generates garbage — so a benchmark records
throughput for a broken model.

| checkpoint format | result |
|---|---|
| plain BF16 | ✅ works |
| `compressed-tensors` / `nvfp4-pack-quantized` | ✅ works (`nvidia/Qwen3-8B-NVFP4`) |
| `compressed-tensors` / `pack-quantized` (int4/w4a16) | ❌ loads, emits `archyCalAMILrousciu…` |
| `fp8` (block-FP8) | ❌ loads, emits `upift如无一言stluanncno源halt` |
| `modelopt`, `modelopt_mixed` (NVIDIA ModelOpt) | ❌ loads, emits `0-1 way/radius housing t[i deる` |

`model_info.precision_compatible()` refuses everything outside the allow-set **before** loading.
That guard keys on the **`(quant_method, format)` pair** from
`checkpoint_quant_fingerprint()` — the method alone is insufficient, because `compressed-tensors` is
the method for both the format that works and the one that produces garbage.

**Every model in the table above still serves correctly under vLLM**, which is how the sibling
projects run them. That is the point of the comparison, not a workaround — and the gap is large:
`nvidia/Qwen3-8B-NVFP4` measures 38.8 tok/s under vLLM (real FP4 tensor cores) versus 5.99 through
the HF path, which decompresses to BF16 every forward pass. Anything that needs a quantized
checkpoint benchmarked belongs in `scripts/capture_vllm_reference.py`, not the HF path.

**A correct format label is not a correct checkpoint.** A converted checkpoint can declare
`nvfp4-pack-quantized` truthfully, pass the guard, load, and still be broken — proven on Gemma-4,
where the weights landed under tensor names transformers does not look for and the model generated
nothing. Acceptance for any new checkpoint is **load AND generate**, with the output inspected: an
ASCII-coherence fraction plus an eyeballed sample. A successful load is not evidence.

## Model catalog

`config/model-lists.json` offers **8 LLMs** (all BF16-native except `nvidia/Qwen3-8B-NVFP4`),
**4 VLM/encoder** entries and **2 CNN**. All resolve on disk.

⚠️ **Three of the eight LLMs are symlinks into other projects or into HF-cache snapshot hashes** —
`Qwen3-0.6B` and `Qwen3-8B` point at `~/gb10-agent-stack/models/`, `Mistral-7B-v0.1` at a
`~/.cache/huggingface` snapshot. A cleanup in another project, or a re-download that changes the
snapshot hash, silently removes a live catalog entry; the UI's only response is a sidebar caption
("N catalog model(s) not on disk"), and that model's measured anchor becomes unreproducible. Check
with `find models -xtype l` before trusting the catalog.

The **vision track currently has no runnable VLM.** All six configured vision entries are *encoders*
(CLIP/ViT/DINO/ResNet/EfficientNet) — `is_vision_language_model()` returns False for every one — so
`vision_inference.benchmark_vlm()` and the image+prompt→text path are present and unreachable. The
two real VLMs on disk (`Nemotron-3.5-Super-VL-120B`, `Nemotron-3.5-Bolt-74B`) are ModelOpt and
refused by the guard above. They do ship their own modeling code with an `auto_map` naming
`AutoModelForImageTextToText`, so they are not blocked by stock transformers' architecture support —
only by the quantization format.
