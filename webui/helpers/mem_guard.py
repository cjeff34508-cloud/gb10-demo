"""Memory guard — estimates model footprint and checks against Dell GB10 usable memory."""

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
    ("mixtral-8x7b",  46.7),   # full weight load despite MoE routing
    ("mixtral",       46.7),
    ("tinyllama-1.1b", 1.1),
    ("tinyllama",      1.1),
    ("phi-4",         14.0),
    ("phi-3",          3.8),
    ("qwen3-8b",       8.0),
    ("qwen3.5-35b-a3b", 35.0),  # MoE: 35B total weights load (3B active)
    ("qwen3-30b-a3b",  30.0),   # MoE: 30B total weights load (3B active)
    ("qwen2.5-72b",   72.0),
    ("qwen2.5-70b",   70.0),
    ("qwen2.5-32b",   32.0),
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
    ("72b", 72.0), ("70b", 70.0), ("35b", 35.0), ("32b", 32.0),
    ("30b", 30.0), ("14b", 14.0), ("13b", 13.0), ("8b",   8.0),
    ("7b",   7.0), ("3b",   3.0), ("1.5b",  1.5),
    ("1.1b", 1.1), ("1b",   1.0),
]


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

    base_overhead    = 0.08
    batch_factor     = min(0.003 * batch_size, 0.20)
    context_factor   = 0.04 * (context_len / 512)
    overhead         = min(base_overhead + batch_factor + context_factor, 0.50)
    return raw_gb * (1.0 + overhead)


def fits_in_memory(
    model_name: str, precision: str, batch_size: int = 1, context_len: int = 256
) -> tuple[bool, float]:
    """Return (fits: bool, estimated_gb: float) for the Dell GB10 usable window."""
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
