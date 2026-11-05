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

# ---------------------------------------------------------------------------
# Per-model measured anchors for the TCO tab
# ---------------------------------------------------------------------------
# The TCO engine never loads a model — it needs catalog metadata, a user count,
# and ONE throughput anchor. That anchor used to come from the last HF benchmark
# run in the session, or a hardcoded 1200 tok/s, applied to every model alike.
# Both are bad: the HF path cannot run any 2026 quantized checkpoint
# (docs/CATALOG-2026.md), and a single number across models that differ 40x in
# active parameters is why $/MTok read identically for all of them.
#
# vLLM captures fix it: they are real single-stream measurements, they exist per
# model, and they do not depend on the HF stack at all. This map ties a
# MODEL_CATALOG key to its capture key.
_CATALOG_TO_CAPTURE: dict[str, str] = {
    "Qwen2.5-7B":              "qwen2.5-7b",
    "nvidia/Qwen3-8B-NVFP4":   "qwen3-8b-nvfp4",
    "Qwen3.6-35B-A3B":         "qwen3.6-35b-a3b-fp8",
    # added as the capture run is extended:
    # "Nemotron-3.5-Lightning-30B-A3B": "nemotron-3.5-lightning-30b-a3b",
}


def measured_tps(catalog_name: str) -> tuple[float | None, str]:
    """``(single_stream_tok_s, provenance)`` for a MODEL_CATALOG entry.

    Returns the capture's n=1 per-user figure — single-stream decode, the same
    quantity the TCO engine transfers across hardware. ``provenance`` is a short
    human string for the UI so a number is never shown without saying where it
    came from.
    """
    key = _CATALOG_TO_CAPTURE.get(catalog_name)
    if not key:
        return None, f"no vLLM capture for {catalog_name}"
    cap = get_capture(key)
    if not cap:
        return None, f"capture '{key}' not in the reference file"
    rows = cap.get("per_n") or []
    row1 = next((r for r in rows if r.get("n_users") == 1), None)
    if not row1:
        return None, f"capture '{key}' has no n=1 level"
    ts = (cap.get("timestamp") or "")[:10]
    return float(row1["per_user_tps"]), (
        f"measured on this box via vLLM — {cap.get('precision','?')}, "
        f"single stream, captured {ts}")


def measured_models() -> list[str]:
    """MODEL_CATALOG keys that have a usable measured anchor."""
    return [m for m in _CATALOG_TO_CAPTURE if measured_tps(m)[0] is not None]
