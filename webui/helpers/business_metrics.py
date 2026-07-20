"""Derive business-relevant output from raw BenchmarkMetrics."""

import torch
from typing import Any, Dict, Optional

# Reference cost basis: H100 SXM5 on-demand ~$3.50/hr
# Used to frame Dell GB10 value — "what would this cost you on cloud?"
_REFERENCE_HOURLY_USD = 3.50


def derive_business_output(metrics, total_vram_mb: float = None) -> Dict[str, Any]:
    """
    Build the business_output dict for a BenchmarkMetrics result.
    Called after latency/memory/tokens_per_sec are populated.
    """
    out: Dict[str, Any] = {}

    if total_vram_mb is None:
        total_vram_mb = (
            torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            if torch.cuda.is_available() else 0
        )

    # Cost per million tokens (H100 cloud equivalent rate)
    tps = metrics.tokens_per_sec
    if tps > 0:
        cost_per_sec = _REFERENCE_HOURLY_USD / 3600.0
        out["cost_per_mtok"] = round((cost_per_sec / tps) * 1_000_000, 4)

    # Max concurrent sessions that fit in VRAM
    if total_vram_mb > 0 and metrics.peak_memory_mb > 0:
        out["max_concurrent_sessions"] = max(1, int(total_vram_mb / metrics.peak_memory_mb))

    # p95 / p99 from raw latency list; fall back to estimates when runs < 20
    lats = getattr(metrics, "_raw_latencies", [])
    if lats:
        s = sorted(lats)
        n = len(s)
        out["p95_ms"] = round(s[max(0, int(n * 0.95) - 1)], 2)
        out["p99_ms"] = round(s[max(0, int(n * 0.99) - 1)], 2)
    elif metrics.latency_ms > 0:
        # Rough statistical estimates when only avg is available
        out["p95_ms"] = round(metrics.latency_ms * 1.15, 2)
        out["p99_ms"] = round(metrics.latency_ms * 1.30, 2)

    # Images/sec (vision workloads)
    name = metrics.model_name.lower()
    is_vision = any(k in name for k in ("clip", "vit", "dino", "vision"))
    if is_vision and metrics.throughput_samples_per_sec > 0:
        out["images_per_sec"] = round(metrics.throughput_samples_per_sec, 1)

    # Jobs/hour (HPC workloads)
    is_hpc = any(k in name for k in ("matmul", "bandwidth", "lob", "reduction", "fill", "concurrent"))
    if is_hpc and metrics.latency_ms > 0:
        out["jobs_per_hour"] = round(3600 / (metrics.latency_ms / 1000))

    return out
