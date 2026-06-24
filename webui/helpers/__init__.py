"""WebUI helpers package."""

from .benchmark_utils import BenchmarkMetrics, benchmark_model_inference
from .llm_inference import LLMInference, benchmark_llm
from .vision_inference import VisionModelInference, benchmark_vision_model
from .hpc_compute import HPCBenchmark, benchmark_hpc
from .memory_stress import MemoryStress
from .bottleneck_analyzer import classify_bottleneck, get_gpu_stats, bottleneck_badge_html, BOTTLENECK_COLORS
from .business_metrics import derive_business_output

__all__ = [
    "BenchmarkMetrics",
    "benchmark_model_inference",
    "LLMInference",
    "benchmark_llm",
    "VisionModelInference",
    "benchmark_vision_model",
    "HPCBenchmark",
    "benchmark_hpc",
    "MemoryStress",
    "classify_bottleneck",
    "get_gpu_stats",
    "bottleneck_badge_html",
    "BOTTLENECK_COLORS",
    "derive_business_output",
]
