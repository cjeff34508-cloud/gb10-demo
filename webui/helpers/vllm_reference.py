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


# ---------------------------------------------------------------------------
# PREDICTED anchors (webui/data/predicted_anchors.json) — the FOURTH tier
# ---------------------------------------------------------------------------
# The 2026 frontier models in MODEL_CATALOG cannot be measured here: every one
# needs 4-32 GB10s just to hold its weights. Before this, each of them inherited
# the 13.1 tok/s fallback — Qwen2.5-7B's July figure — so the TCO tab priced a
# 2.8T model exactly like a 7B. Predicting openly is better than inheriting a
# wrong number silently, PROVIDED the prediction is visually distinct from a
# measurement and carries its band. That is the whole contract of this tier.
#
# Decode on this box is memory-bandwidth-bound, so:
#     tok/s = bandwidth / (active_b * bytes_per_param) * efficiency
# `efficiency` is DERIVED from the four on-box measurements at call time, never
# hardcoded, and the leave-one-out back-test in scripts/smoke_tco_capacity.py
# must retrodict all four before any of these numbers is allowed to publish.
_PRED_PATH = _REF_PATH.parent / "predicted_anchors.json"


# Largest leave-one-out relative error the predictor is allowed to have before
# its numbers stop being publishable. Pinned by an assertion in
# scripts/smoke_tco_capacity.py, which is what makes this a gate and not a knob:
# add a calibration row that makes the model worse and the suite goes red.
_PRED_MAX_LOO_ERR = 0.35


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _calibration_rows() -> tuple[float, list[tuple[float, float]]] | None:
    """``(bandwidth_gbs, [(gb_per_token, measured_tps), ...])``."""
    try:
        d = json.loads(_PRED_PATH.read_text())
    except Exception:
        return None
    bw = float(d.get("bandwidth_gbs") or 0)
    rows = []
    for r in d.get("calibration") or []:
        gb = float(r.get("active_b") or 0) * float(r.get("bytes_per_param") or 0)
        tps = float(r.get("measured_tps") or 0)
        if gb > 0 and tps > 0:
            rows.append((gb, tps))
    if bw <= 0 or len(rows) < 2:
        return None
    return bw, rows


def loo_errors() -> list[float] | None:
    """Signed leave-one-out relative errors of the predictor on its own data.

    Calibrate on every row but one, predict the held-out row, and report
    ``(pred - actual) / actual``. This is the predictor's only honest accuracy
    statement, and the published band comes from it.
    """
    cal = _calibration_rows()
    if not cal:
        return None
    bw, rows = cal
    out = []
    for i, (gb, tps) in enumerate(rows):
        rest = [rows[j] for j in range(len(rows)) if j != i]
        med = _median([t / (bw / g) for g, t in rest])
        out.append(((bw / gb) * med - tps) / tps)
    return out


def _calibration() -> tuple[float, float, float, int] | None:
    """``(bandwidth_gbs, eff_median, band_frac, n)`` from the calibration set.

    ``band_frac`` is the largest ABSOLUTE leave-one-out error, so the band we
    publish is the predictor's measured out-of-sample accuracy.

    It is deliberately NOT the in-sample efficiency spread. That spread
    (0.446-0.672 over four points) is narrower than the real error and, worse,
    a min-max interval over n-1 points structurally cannot cover a held-out
    endpoint — holding out the best or worst measurement fails by construction,
    which says nothing about the model. The leave-one-out error does say
    something: it is +/-27% here, against a +/-18% in-sample spread.
    """
    cal = _calibration_rows()
    if not cal:
        return None
    bw, rows = cal
    med = _median([tps / (bw / gb) for gb, tps in rows])
    errs = loo_errors() or []
    band = max((abs(e) for e in errs), default=0.0)
    return bw, med, band, len(rows)


def effective_bytes_per_param(catalog_name: str, precision: str | None = None) -> float | None:
    """Bytes per parameter from the MEASURED checkpoint size, not a nominal constant.

    Every 2026 4-bit release is mixed-precision — NVFP4 on the expert
    projections, BF16 or FP8 everywhere else — so none of them is actually
    0.5 B/param; the six frontier entries run 0.54-1.95. Where ``disk_gb``
    carries a real checkpoint size we divide it by total params; otherwise we
    fall back to the engine's nominal table and say so via the caller.
    """
    from . import tco_engine as _T
    info = _T.MODEL_CATALOG.get(catalog_name) or {}
    p_b = info.get("params_b")
    if not p_b:
        return None
    prec = (precision or _T.native_precision(catalog_name) or "").upper()
    disk = info.get("disk_gb")
    if isinstance(disk, dict) and disk.get(prec):
        return float(disk[prec]) / float(p_b)
    return _T._BYTES_PER_PARAM.get(prec)


def predicted_anchor(
    catalog_name: str, precision: str | None = None
) -> tuple[float | None, str, tuple[float, float] | None]:
    """``(tps, provenance, (lo, hi))`` — a PREDICTION, never a measurement.

    Returns ``(None, why, None)`` whenever the inputs are not good enough to
    predict from, which is the honest outcome: an un-anchored model should show
    "enter a number" rather than a figure the data cannot support.
    """
    cal = _calibration()
    if not cal:
        return None, "no usable calibration set in predicted_anchors.json", None
    bw, med, band, n = cal
    if band > _PRED_MAX_LOO_ERR:
        return None, (f"predictor rejected: leave-one-out error {band:.0%} exceeds the "
                      f"{_PRED_MAX_LOO_ERR:.0%} ceiling — enter a number instead"), None

    from . import tco_engine as _T
    info = _T.MODEL_CATALOG.get(catalog_name) or {}
    if not info:
        return None, f"{catalog_name} is not in MODEL_CATALOG", None
    # The model is autoregressive-decode-specific: a vision ENCODER has no
    # tokens/sec at all, so predicting one for CLIP/ViT/ResNet would be a
    # category error dressed up as a figure. Refuse instead.
    if info.get("category") != "LLM" or info.get("type") not in ("moe", "decoder"):
        return None, (f"{catalog_name} is a {info.get('category','?')} "
                      f"{info.get('type','?')} — decode tok/s does not apply"), None
    # Bandwidth per token follows ACTIVE parameters for an MoE (that is the whole
    # reason a 1.6T model can out-run a 550B one) and total params for a dense.
    act = info.get("active_b") if info.get("type") == "moe" else info.get("params_b")
    act = act or info.get("params_b")
    bpp = effective_bytes_per_param(catalog_name, precision)
    if not act or not bpp:
        return None, f"no active-param / bytes-per-param basis for {catalog_name}", None

    gb_per_tok = float(act) * float(bpp)
    ceiling = bw / gb_per_tok
    prec = (precision or _T.native_precision(catalog_name) or "?").upper()
    basis = "measured checkpoint size" if isinstance(info.get("disk_gb"), dict) \
        and info["disk_gb"].get(prec) else "nominal bytes/param"
    point = ceiling * med
    why = (
        f"PREDICTED — not measured, ±{band:.0%}. {bw:g} GB/s ÷ {gb_per_tok:.1f} GB/token "
        f"({act:g}B active × {bpp:.3f} B/param, {basis}) × {med:.3f} efficiency, "
        f"calibrated on {n} single-stream measurements taken on this box. The ±{band:.0%} "
        f"is the predictor's largest leave-one-out error against those same "
        f"measurements, not a guess. Moves $/MTok and aggregate throughput only — "
        f"capacity, node count and CapEx come from attention geometry and measured "
        f"checkpoint size, and do not depend on this figure."
    )
    return point, why, (point * (1.0 - band), point * (1.0 + band))


def anchor_with_prediction(
    catalog_name: str,
) -> tuple[float | None, str, str, tuple[float, float] | None]:
    """``(tps, provenance, tier, band)`` across the full four-tier ladder.

    ``tier`` is one of ``"vllm"``, ``"hf"``, ``"predicted"``, ``"none"`` and is
    what the UI keys its colour off, so a prediction can never be painted as a
    measurement.
    """
    t, why = measured_tps(catalog_name)
    if t:
        return t, why, "vllm", None
    t, why = _hf_anchor(catalog_name)
    if t:
        return t, why, "hf", None
    t, why, band = predicted_anchor(catalog_name)
    if t:
        return t, why, "predicted", band
    return None, why, "none", None


def predicted_models() -> list[str]:
    """Catalog keys that have NO measurement but CAN be predicted."""
    from . import tco_engine as _T
    done = set(anchored_models())
    return [m for m in _T.MODEL_CATALOG
            if m not in done and predicted_anchor(m)[0] is not None]
