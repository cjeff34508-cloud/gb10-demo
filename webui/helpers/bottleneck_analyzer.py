"""Bottleneck classification using pynvml + observed metrics."""

import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False

# Dell Pro Max GB10 real peaks
_GB10_BW_GBS = 273       # LPDDR5X unified-memory bandwidth (~273 GB/s), not NVLink-C2C / "4 TB/s"
_GB10_TFLOPS_FP16 = 500  # ~500 TFLOPS FP16 dense (1 PFLOP is the sparse FP4 figure)


def get_gpu_stats() -> dict:
    """Return live GPU utilization, memory, and power via pynvml."""
    if not _NVML:
        return {}
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(h)
        return {
            "gpu_util_pct": util.gpu,
            "mem_util_pct": util.memory,
            "mem_used_mb": mem.used / 1024 / 1024,
            "mem_total_mb": mem.total / 1024 / 1024,
            "power_w": round(power_mw / 1000, 1),
        }
    except Exception:
        return {}


def classify_bottleneck(metrics, total_vram_mb: float = None) -> str:
    """
    Classify primary bottleneck from observed metrics.

    Priority order:
      1. Memory Capacity  — VRAM utilization > 85%
      2. Storage/Tiering  — load time dominates run time
      3. Memory Bandwidth — bandwidth-bound workload near peak
      4. GPU Compute       — high GPU utilisation, not bandwidth saturated
      5. CPU-GPU Orch.    — fallback for small models / overhead-dominated
    """
    if total_vram_mb is None:
        total_vram_mb = (
            torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            if torch.cuda.is_available() else 0
        )

    # 1. Memory capacity
    if total_vram_mb > 0 and metrics.peak_memory_mb > 0:
        if metrics.peak_memory_mb / total_vram_mb > 0.85:
            return "Memory Capacity"

    # 2. Storage / tiering — load time > 2x the inference work it enables.
    #    The weight load is a ONE-TIME cost amortised across every session the
    #    loaded copy then serves, so it must be compared against the total
    #    inference work, not one request's latency. Without the batch term a
    #    32-session concurrency run on a small model came back "Storage /
    #    Tiering" purely because a 10.6 s cold load beat a 1.5 s request -- which
    #    says nothing about the steady-state serving the test exists to measure.
    #    At batch 1 (a single cold query) the two are identical, so the original
    #    verdict is preserved exactly where it was right.
    load_ms = metrics.business_output.get("load_ms", 0)
    _bs = (metrics.operational_condition or {}).get("batch_size", 1) or 1
    _work_ms = metrics.latency_ms * max(1, int(_bs))
    if load_ms > 0 and _work_ms > 0 and load_ms > _work_ms * 2:
        return "Storage / Tiering"

    # 3. Kernel-intrinsic HPC classification — these workloads are bottlenecked by
    #    the resource their kernel is built around, regardless of the absolute number
    #    (a streaming/reduction kernel is memory-bound even at low GB/s on a busy box).
    name = metrics.model_name.lower()
    is_bw_test = any(k in name for k in
                     ("bandwidth", "lob", "fill", "reduction", "memory", "monte carlo"))
    if is_bw_test:
        return "Memory Bandwidth"
    if any(k in name for k in ("matmul", "scholes", "black-scholes")):
        return "GPU Compute"   # dense matmul / transcendental pricing = compute-bound

    # 4. pynvml live stats (model inference)
    stats = get_gpu_stats()
    if stats:
        if stats.get("gpu_util_pct", 0) >= 80:
            return "GPU Compute"
        if stats.get("mem_util_pct", 0) >= 60:
            return "Memory Bandwidth"

    # 5. Heuristic: LLM decode is almost always memory-bandwidth bound
    is_llm = any(k in name for k in (
        "llama", "qwen", "mistral", "phi", "tinyllama", "mixtral", "gpt", "nvidia"
    ))
    if is_llm and metrics.tokens_per_sec > 0:
        return "Memory Bandwidth"

    return "GPU Compute"


# Badge colours for Streamlit markdown
BOTTLENECK_COLORS = {
    "Memory Capacity":       "#dc3545",   # red
    "Memory Bandwidth":      "#fd7e14",   # orange
    "GPU Compute":           "#28a745",   # green (utilised = good)
    "CPU-GPU Orchestration": "#17a2b8",   # teal
    "Storage / Tiering":     "#6c757d",   # grey
}


def bottleneck_badge_html(label: str) -> str:
    color = BOTTLENECK_COLORS.get(label, "#6c757d")
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.85em;font-weight:600;">{label}</span>'
    )
