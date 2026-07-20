"""
Benchmark utilities for precision comparison across models.
Handles timing, memory profiling, and quantization support.
"""

import gc
import time
from contextlib import contextmanager
from functools import wraps
from typing import Dict, Tuple, Any, Optional, List

import torch
import psutil


def free_cuda_memory() -> None:
    """Release Python references and return cached GPU blocks to the driver.

    Call after any benchmark that allocated large tensors so `nvidia-smi`/top
    reflect the freed memory before the next run. Safe to call on CPU-only hosts.
    """
    gc.collect()
    if torch.cuda.is_available():
        # This is a teardown utility (called from unload_narrator_for_benchmark and
        # post-benchmark cleanup). Guard both calls so a context already in an error
        # state — e.g. a prior device-side assert during generation — can't crash the
        # app here; the original error still surfaces where it actually happened.
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


class BenchmarkMetrics:
    """Container for benchmark results across all 4 failure-mode dimensions."""

    def __init__(self, model_name: str, precision: str):
        self.model_name = model_name
        self.precision = precision
        # The precision the model ACTUALLY ran at, which can differ from the
        # requested one — e.g. FP16 on a BF16-native model runs as BF16 (a raw
        # FP16 cast overflows BF16-trained activations → NaN/garbage). precision_note
        # explains any such substitution for honest display.
        self.effective_precision: str = precision
        self.precision_note: str = ""
        # Core perf
        self.latency_ms: float = 0.0
        self.memory_mb: float = 0.0
        self.peak_memory_mb: float = 0.0
        self.throughput_samples_per_sec: float = 0.0
        self.tokens_per_sec: float = 0.0
        self.error: Optional[str] = None
        # Dimension 1: Workload phase
        self.workload_phase: str = ""
        # Dimension 2: Primary bottleneck
        self.primary_bottleneck: str = ""
        # Dimension 3: Operating condition
        self.operational_condition: Dict[str, Any] = {}
        # Dimension 4: Business-relevant output
        self.business_output: Dict[str, Any] = {}
        # Raw latencies for p95/p99
        self._raw_latencies: List[float] = []

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "model": self.model_name,
            "precision": self.precision,
            "latency_ms": round(self.latency_ms, 3),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "throughput": round(self.throughput_samples_per_sec, 2),
            "tokens_per_sec": round(self.tokens_per_sec, 1) if self.tokens_per_sec > 0 else None,
            "error": self.error,
            # Dimension 1
            "workload_phase": self.workload_phase,
            # Dimension 2
            "primary_bottleneck": self.primary_bottleneck,
        }
        # Dimension 3 — flatten operational_condition
        for k, v in self.operational_condition.items():
            d[f"cond_{k}"] = v
        # Dimension 4 — flatten business_output
        for k, v in self.business_output.items():
            d[f"biz_{k}"] = v
        return d


@contextmanager
def gpu_memory_context(stats: Optional[Dict[str, float]] = None):
    """Track GPU memory allocation across a block.

    Pass a dict to receive the measurements (a generator-based context manager
    cannot ``return`` a value to the caller, so results are written into ``stats``):

        info = {}
        with gpu_memory_context(info):
            ...
        print(info["peak_mb"])
    """
    if not torch.cuda.is_available():
        yield
        return
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated() / 1024 / 1024
    try:
        yield
    finally:
        memory_after = torch.cuda.memory_allocated() / 1024 / 1024
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        if stats is not None:
            stats["allocated_mb"] = memory_after
            stats["delta_mb"] = memory_after - memory_before
            stats["peak_mb"] = peak_memory


def measure_latency(func):
    """Decorator to measure function execution latency."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        return result, elapsed
    return wrapper


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_gpu_peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def get_total_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    return 0.0


def quantize_to_precision(tensor: torch.Tensor, precision: str) -> torch.Tensor:
    """Quantize tensor to specified precision."""
    if precision == "FP32":
        return tensor.float()
    elif precision == "FP16":
        return tensor.half()
    elif precision == "BF16":
        return tensor.to(torch.bfloat16)
    elif precision in ("INT8", "FP8"):
        scale = 127.0 / tensor.abs().max().clamp(min=1e-5)
        quantized = (tensor * scale).round().clamp(-127, 127)
        return quantized / scale
    elif precision in ("FP4", "NVFP4"):
        # FP4/NVFP4: 4-bit, 16 quantization levels
        scale = 7.0 / tensor.abs().max().clamp(min=1e-5)
        quantized = (tensor * scale).round().clamp(-7, 7)
        return quantized / scale
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def create_dummy_input(input_type: str, batch_size: int = 1) -> torch.Tensor:
    """Create dummy input tensor for benchmarking."""
    if input_type == "text":
        return torch.randint(0, 1000, (batch_size, 32)).cuda() if torch.cuda.is_available() else torch.randint(0, 1000, (batch_size, 32))
    elif input_type == "image":
        return torch.randn(batch_size, 3, 224, 224).cuda() if torch.cuda.is_available() else torch.randn(batch_size, 3, 224, 224)
    elif input_type == "tensor":
        return torch.randn(1000, 1000).cuda() if torch.cuda.is_available() else torch.randn(1000, 1000)
    else:
        raise ValueError(f"Unknown input type: {input_type}")


def benchmark_model_inference(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    num_runs: int = 3,
    warmup_runs: int = 1,
) -> Tuple[float, float]:
    """Benchmark model inference latency and memory. Returns (latency_ms, peak_memory_mb)."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_data)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    avg_latency = sum(latencies) / len(latencies)
    peak_memory = get_gpu_peak_memory_mb()
    return avg_latency, peak_memory


def format_results(metrics: BenchmarkMetrics, reference_metrics: Optional[BenchmarkMetrics] = None) -> str:
    lines = [
        f"Model: {metrics.model_name}",
        f"Precision: {metrics.precision}",
        f"Latency: {metrics.latency_ms:.2f} ms",
        f"Memory: {metrics.memory_mb:.1f} MB",
        f"Peak Memory: {metrics.peak_memory_mb:.1f} MB",
    ]
    if reference_metrics and metrics.latency_ms > 0:
        speedup = reference_metrics.latency_ms / metrics.latency_ms
        memory_ratio = metrics.memory_mb / reference_metrics.memory_mb if reference_metrics.memory_mb > 0 else 0
        lines.append(f"Speedup vs {reference_metrics.precision}: {speedup:.2f}x")
        lines.append(f"Memory vs {reference_metrics.precision}: {memory_ratio:.2f}x")
    if metrics.error:
        lines.append(f"Error: {metrics.error}")
    return "\n".join(lines)
