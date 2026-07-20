# Dell GB10 Demo Suite

A sales-demo benchmark suite for the **Dell GB10 (Grace-Blackwell, aarch64)**.
It benchmarks LLM / VLM / CNN / HPC workloads across precisions (FP4, FP8, FP16, FP32,
FP64) and surfaces **business-relevant output** — bottleneck classification, TCO, and
hardware-fit recommendations — in a Streamlit UI organized **by failure-mode scenario**,
not by modality.

> The canonical engineering spec is [`CLAUDE.md`](CLAUDE.md).

## Highlights

- **Phase-separated benchmarking** — load / prefill / decode timed independently, with
  warm-cache fairness so per-precision load times are comparable.
- **Precision sweep** — FP4 / NVFP4 / FP8 / INT8 / FP16 / BF16 / FP32 via bitsandbytes +
  native dtypes, with honest notes on what the Dell GB10's HF stack actually computes (e.g. NVFP4
  decompresses to BF16 — see `CLAUDE.md`).
- **TCO engine** (`webui/helpers/tco_engine.py`) — Dell PowerEdge XE catalog, per-GPU
  spec sheet (`GPU_SPECS`), bandwidth-contention and fleet-coordination modeling, and a
  hardware-fit recommendation. Two rating profiles:
  - **LLM Inference** — per-user tok/s + $/MTok, anchored on *measured* Dell GB10 throughput.
  - **FinTech / Bandwidth-bound** — sizes a **Monte-Carlo** workload (resident paths ×
    bytes/path, timesteps, target paths/sec) to the fewest GPUs that fit the working set
    and meet the demanded memory bandwidth; ranks Dell platforms by $/B-paths.
- **On-device AI narrator** (`webui/helpers/on_device_ai.py`) — a local instruct model
  that narrates results and the TCO recommendation *on the Dell GB10*, no cloud.

## Layout

| Path | What |
|------|------|
| `setup.sh` | One-time host setup — verifies CUDA, installs cuDNN/TensorRT, builds the 4 venvs. Writes `.setup-state`. |
| `config/` | `model-lists.json` catalog + per-workload `requirements-*.txt`. |
| `webui/` | The Streamlit app (`streamlit_app.py`) + `helpers/` (TCO, inference, narrator, bottleneck analyzer, mem guard). |
| `workloads/` | Standalone `hpc/ llm/ vlm/ cnn` workload scripts. |
| `scripts/` | `model-downloader.py` (HF pull per config), `convert_nvfp4_to_ct.py`, etc. |

## Requirements

- Ubuntu 24.04 LTS **aarch64**, **Dell GB10**, CUDA 13.0
- Python 3.12+, cuDNN 9.13, TensorRT 11.0
- 128 GB unified memory (CPU+GPU share the pool; ~100 GB usable)

> **Not included in this repo:** model weights (`models/`, ~368 GB) and the Python venvs
> (`env/`, ~24 GB). Both are regenerated — see Quick start.

## Quick start

```bash
bash setup.sh                          # build the venvs (env/{hpc,llm,vlm,cnn})
python scripts/model-downloader.py     # pull weights per config/model-lists.json
                                       # (HF token at ~/.cache/huggingface/token or ~/hf-token)
bash webui/launch.sh                   # launch the UI → http://localhost:8501
```

Each workload is venv-isolated:
```bash
source env/<workload>/bin/activate     # hpc | llm | vlm | cnn
```

## Notes

- Precision support in the LLM path is via bitsandbytes (FP4/FP8) + native FP16/BF16/FP32.
- Keep new UI logic in `webui/helpers/`; `streamlit_app.py` orchestrates, helpers compute.
- HF downloads are resumable; the token is read at runtime (never committed).

## License

No license is granted. Internal demo material — confirm with the maintainer before redistributing.
