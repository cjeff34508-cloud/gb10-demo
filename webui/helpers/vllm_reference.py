"""
Loader for the captured vLLM production-reference data.

The demo NEVER runs vLLM in-process (one heavy GPU workload at a time, and vLLM
lives only in the qwen-hermes Docker container). Instead, an operator runs
``scripts/capture_vllm_reference.py`` offline against the running container, which
writes ``webui/data/vllm_reference.json``. The Streamlit app only *reads* that file
here — so a missing file simply means "no real serving data captured yet" and the UI
degrades honestly rather than fabricating numbers.

JSON shape (produced by the capture script):
    {
      "captures": {
        "<key>": {                      # e.g. "qwen2.5-7b", "qwen3-8b-nvfp4"
          "model": str, "served_name": str, "precision": str,
          "gpu": str, "timestamp": str,
          "per_n": [                    # one row per concurrency level
            {"n_users": int, "per_user_tps": float, "aggregate_tps": float,
             "kv_cache_pct": float, "gpu_util": float}
          ],
          "hf_eager_tps": float|null,   # matched single-user HF-eager tok/s (for the delta)
          "fp4_supported": bool|null,   # only set for the NVFP4 capture
          "error": str|null,
          "notes": str
        }
      }
    }
"""

import json
from pathlib import Path

_REF_PATH = Path(__file__).resolve().parent.parent / "data" / "vllm_reference.json"


def reference_path() -> Path:
    """Absolute path to the reference JSON (may not exist)."""
    return _REF_PATH


def load_vllm_reference() -> dict | None:
    """Return the parsed reference dict, or None if absent/empty/invalid."""
    try:
        if not _REF_PATH.exists():
            return None
        data = json.loads(_REF_PATH.read_text())
        return data or None
    except Exception:
        return None


def get_capture(key: str) -> dict | None:
    """Return one capture block by key (e.g. 'qwen2.5-7b'), or None."""
    data = load_vllm_reference()
    if not data:
        return None
    return (data.get("captures") or {}).get(key)


def has_real_fp4() -> bool:
    """True only when an NVFP4 capture exists AND vLLM reported real FP4 support."""
    data = load_vllm_reference()
    if not data:
        return False
    for cap in (data.get("captures") or {}).values():
        if cap.get("fp4_supported") is True and (cap.get("per_n") or cap.get("hf_eager_tps")):
            return True
    return False
