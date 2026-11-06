"""Memory guard — estimates model footprint and checks against Dell Pro Max GB10 usable memory."""

GB10_TOTAL_GB   = 128.0
GB10_RESERVE_GB = 28.0
GB10_USABLE_GB  = GB10_TOTAL_GB - GB10_RESERVE_GB  # 100 GB

_BYTES_PER_PARAM: dict[str, float] = {
    "FP64":  8.0,
    "FP32":  4.0,
    "FP16":  2.0,
    "BF16":  2.0,
    "FP8":   1.0,
    "INT8":  1.0,
    "FP4":   0.5,
    "NVFP4": 0.5,
}

# Known model param counts (B) — keyed by lowercase substring of model name
_MODEL_PARAMS_B: list[tuple[str, float]] = [
    # LLM — specific first, then generic size suffixes
    # --- 2026 catalog (C2, 2026-11-05). Explicit entries, because the generic
     # suffix fallback below is substring-based and gets these badly wrong:
    # "qwen3.8-27b" matches "7b" (27 -> 7B) and "laguna"/"120b" match nothing
    # at all, falling through to the 7.0 default. A 118B checkpoint reported as
    # 7B passes the fits-in-memory check and the UI then tries to load it,
    # which is exactly the unified-memory over-commit that hung the box on
    # 2026-06-30. Keep these FIRST.
    ("laguna-s-2.1",                118.0),  # 92.9 GiB on disk (mixed NVFP4/BF16)
    ("laguna",                      118.0),
    ("nemotron-3-super-120b",       120.0),
    ("nemotron-3.5-super-vl-120b",  120.0),
    ("nemotron-3.5-bolt-74b",        74.0),
    ("nemotron-3.5-lightning-30b",   30.0),
    ("gemma-4-26b",                  26.0),
    ("qwen3.6-35b-a3b",              35.0),
    ("qwen3.8-27b",                  27.0),
    # Plain BF16 additions (2026-11-05). Explicit, because the generic suffix
    # table has no "0.6b" key and "qwen3-0.6b" fell through to the 7.0 default
    # — a 0.6B model reported as 7B.
    ("qwen3-0.6b",                    0.6),
    ("qwen3-8b",                      8.0),
    ("qwen2.5-32b",                  32.0),
    ("mixtral-8x7b",  46.7),   # full weight load despite MoE routing
    ("mixtral",       46.7),
    ("tinyllama-1.1b", 1.1),
    ("tinyllama",      1.1),
    ("phi-4",         14.0),
    ("phi-3",          3.8),
    ("qwen3-30b-a3b",  30.0),   # MoE: 30B total weights load (3B active)
    ("qwen2.5-72b",   72.0),
    ("qwen2.5-70b",   70.0),
    ("qwen2.5-14b",   14.0),
    ("qwen2.5-7b",     7.0),
    ("llama-3.3-70b", 70.0),
    ("llama-3.2-1b",   1.0),
    ("mistral-7b",     7.0),
    # VLM (weights are small)
    ("clip-vit-large", 0.43),
    ("clip-vit-base",  0.15),
    ("vit-base",       0.086),
    ("dino-vits16",    0.021),
    # Generic size suffixes (last resort) — larger/longer keys first so e.g.
    # "35b" matches before "3b" (which also appears in "...-a3b").
    ("120b", 120.0), ("118b", 118.0), ("74b", 74.0), ("72b", 72.0),
    ("70b", 70.0), ("35b", 35.0), ("32b", 32.0), ("27b", 27.0), ("26b", 26.0),
    ("30b", 30.0), ("14b", 14.0), ("13b", 13.0), ("8b",   8.0),
    ("7b",   7.0), ("3b",   3.0), ("1.5b",  1.5),
    ("1.1b", 1.1), ("1b",   1.0), ("0.6b", 0.6), ("0.5b", 0.5),
]


# Measured on-disk checkpoint size (GB), for MIXED-precision releases where
# params_b x bytes_per_param is wrong. NVFP4 releases commonly quantize only the
# MoE expert projections and leave the rest BF16, so the real file is far larger
# than params x 0.5. Under-stating the footprint is the DANGEROUS direction on a
# 121 GB unified pool, so a measurement always wins over the flat estimate.
# Mirrors MODEL_CATALOG["disk_gb"] in tco_engine.py — keep the two in step.
#
# ⚠️ Valid ONLY for a checkpoint that will load *as* quantized. If transformers
# does not implement the checkpoint's quantization it prints "we will skip the
# quantization" and materialises the model at full width, so the on-disk size
# becomes a LOWER BOUND, not an estimate. Measured 2026-11-06: the 75.1 GB
# Nemotron-3.5-Super-VL reached 146 GB total-vm that way and triggered a global
# OOM-kill. That is why precision_compatible() refuses an unsupported
# quant_method BEFORE loading — the refusal is a memory-safety property, not
# just a correctness one.
_MEASURED_GB: list[tuple[str, float]] = [
    ("laguna-s-2.1",                 99.8),   # 92.9 GiB measured; flat FP4 said 59.0
    ("nemotron-3-super-120b",        80.4),   # measured; flat FP4 said 60.0
    # The VL checkpoints were missing, and the Super-VL one was UNDER-stated:
    # estimated 66.2 GB against a real 75.1 GB file, so the guard would have
    # waved through a load 9 GB larger than it believed.
    ("nemotron-3.5-super-vl-120b",   75.1),   # measured 2026-11-05
    ("nemotron-3.5-bolt-74b",        24.1),   # measured; flat FP4 said 37.0 (over-stated)
    # Also UNDER-stated, found 2026-11-06. Same mixed-precision release pattern:
    # only part of the checkpoint is NVFP4, so the flat 0.5 byte/param estimate
    # comes in below the real file. Both are refused by precision_compatible
    # today (ModelOpt), but the guard must not believe them smaller than they are.
    ("nemotron-3.5-lightning-30b",   21.6),   # measured; flat FP4 said 16.5
    ("gemma-4-26b",                  18.8),   # measured; flat FP4 said 14.3
]


def measured_gb(model_name: str) -> float | None:
    """Measured on-disk size (GB) for this checkpoint, or None if unmeasured."""
    n = model_name.lower()
    for key, val in _MEASURED_GB:
        if key in n:
            return val
    return None


def params_b(model_name: str) -> float:
    """Estimate parameter count (billions) from model name."""
    n = model_name.lower()
    for key, val in _MODEL_PARAMS_B:
        if key in n:
            return val
    return 7.0  # safe default


def estimate_model_gb(
    model_name: str, precision: str, batch_size: int = 1, context_len: int = 256
) -> float:
    """
    Estimate peak GPU memory (GB) for a model + precision + runtime conditions.

    Overhead scales with batch size and context length to account for KV cache growth:
      - base 8 % : framework buffers, activations at batch=1 short context
      - +0.3 % per unit of batch_size  (capped at +20 %)
      - +4 % per 512 tokens of context (uncapped)
      - total overhead capped at 50 %
    """
    p      = params_b(model_name)
    bpp    = _BYTES_PER_PARAM.get(precision.split()[0].upper(), 4.0)
    raw_gb = p * bpp
    # A measured checkpoint beats the flat bytes-per-param figure. Only applied
    # for the quantized precisions the measurement was taken at — asking for
    # BF16 on a pre-quantized NVFP4 release is a precision-compatibility error,
    # handled elsewhere, and must not silently inherit the FP4 file size.
    _m = measured_gb(model_name)
    if _m is not None and precision.split()[0].upper() in ("FP4", "NVFP4"):
        raw_gb = _m

    base_overhead    = 0.08
    batch_factor     = min(0.003 * batch_size, 0.20)
    context_factor   = 0.04 * (context_len / 512)
    overhead         = min(base_overhead + batch_factor + context_factor, 0.50)
    return raw_gb * (1.0 + overhead)


def fits_in_memory(
    model_name: str, precision: str, batch_size: int = 1, context_len: int = 256
) -> tuple[bool, float]:
    """Return (fits: bool, estimated_gb: float) for the Dell Pro Max GB10 usable window."""
    est = estimate_model_gb(model_name, precision, batch_size, context_len)
    return est <= GB10_USABLE_GB, est


def check_precisions(
    model_name: str, precision_list: list[str], batch_size: int = 1, context_len: int = 256
) -> dict[str, dict]:
    """
    Returns per-precision status dict:
      {
        "FP32": {"fits": False, "est_gb": 143.4,
                 "warn": "~143 GB — exceeds 100 GB usable (28 GB reserved)"},
        "BF16": {"fits": True,  "est_gb": 71.7,  "warn": None},
        ...
      }
    """
    result = {}
    for p in precision_list:
        ok, est = fits_in_memory(model_name, p, batch_size, context_len)
        result[p] = {
            "fits":   ok,
            "est_gb": round(est, 1),
            "warn":   None if ok else (
                f"~{est:.0f} GB estimated — exceeds {GB10_USABLE_GB:.0f} GB usable "
                f"({GB10_RESERVE_GB:.0f} GB reserved for OS/runtime)"
            ),
        }
    return result
