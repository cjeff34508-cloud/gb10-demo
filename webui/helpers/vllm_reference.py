"""
Loader for the captured vLLM production-reference data.

The demo NEVER runs vLLM in-process (one heavy GPU workload at a time, and vLLM
runs in a separate container). Instead, an operator runs
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
    if not row1.get("per_user_tps"):
        return None, f"capture '{key}' n=1 row has no per_user_tps"
    ts = (cap.get("timestamp") or "")[:10]
    return float(row1["per_user_tps"]), (
        f"measured on this box via vLLM — {cap.get('precision','?')}, "
        f"single stream, captured {ts}")


def measured_models() -> list[str]:
    """MODEL_CATALOG keys that have a usable measured anchor."""
    return [m for m in _CATALOG_TO_CAPTURE if measured_tps(m)[0] is not None]

# ---------------------------------------------------------------------------
# HF-measured anchors (webui/data/hf_anchors.json)
# ---------------------------------------------------------------------------
# vLLM captures cover the quantized checkpoints the HF stack cannot run. The
# plain checkpoints it CAN run need an anchor too, or they all fall back to one
# number and their $/MTok columns come out identical — a missing measurement
# masquerading as a result. scripts/measure_hf_anchors.py fills this file.
_HF_ANCHORS = _REF_PATH.parent / "hf_anchors.json"


def _hf_anchor(catalog_name: str) -> tuple[float | None, str]:
    try:
        d = json.loads(_HF_ANCHORS.read_text())
    except Exception:
        return None, "no hf_anchors.json"
    a = (d.get("anchors") or {}).get(catalog_name)
    if not a or not a.get("tok_s"):
        return None, f"no HF anchor for {catalog_name}"
    return float(a["tok_s"]), (
        f"measured on this box via HF eager — {a.get('precision','?')}, "
        f"single stream, {str(a.get('timestamp',''))[:10]}")


def best_anchor(catalog_name: str) -> tuple[float | None, str]:
    """Best available per-model single-stream anchor, vLLM first.

    vLLM wins when both exist: it is the stack the box actually serves with, and
    HF eager is slower than batched serving on the same hardware. The caller
    shows the provenance string, so a figure is never displayed without saying
    which stack produced it.
    """
    t, why = measured_tps(catalog_name)
    if t:
        return t, why
    return _hf_anchor(catalog_name)


def anchored_models() -> list[str]:
    """Catalog keys with ANY measured anchor (vLLM or HF)."""
    out = list(measured_models())
    try:
        d = json.loads(_HF_ANCHORS.read_text())
        for k, v in (d.get("anchors") or {}).items():
            if v.get("tok_s") and k not in out:
                out.append(k)
    except Exception:
        pass
    return out


def capture_for_model(model_ref: str) -> tuple[str | None, dict | None]:
    """``(capture_key, capture)`` for the model the user actually benchmarked.

    Accepts an HF repo id (what the Benchmark tab holds) or a MODEL_CATALOG key
    (what the TCO tab holds), and resolves HF id -> catalog key through
    ``tco_engine.BENCHMARKABLE`` before the capture map.

    Exists because the serving-reference and production-delta panels used
    ``next(iter(captures))``, i.e. whichever capture happened to be first in the
    file. Benchmarking the NVFP4 8B then showed a July BF16 Qwen2.5-7B serving
    table, and the two November captures could never be reached at all.
    Returns (None, None) when this model has no capture — the caller must then
    say so rather than substitute another model's measurement.
    """
    if not model_ref:
        return None, None
    caps = (load_vllm_reference() or {}).get("captures") or {}

    cands = [model_ref]
    try:
        from .tco_engine import BENCHMARKABLE
        mapped = BENCHMARKABLE.get(model_ref)
        if mapped:
            cands.append(mapped)
    except Exception:
        pass

    for c in cands:
        key = _CATALOG_TO_CAPTURE.get(c)
        if key and key in caps:
            return key, caps[key]
    # last resort: the model_ref IS a capture key
    if model_ref in caps:
        return model_ref, caps[model_ref]
    return None, None
