"""Memory stress testing — configurable N-user concurrency and spill detection."""

import time
import torch
from .benchmark_utils import (
    BenchmarkMetrics, get_gpu_peak_memory_mb, get_total_vram_mb, free_cuda_memory,
)


def _tag_memory(metrics: BenchmarkMetrics, phase: str, num_users: int, total_vram: float):
    from .bottleneck_analyzer import classify_bottleneck, get_gpu_stats
    from .business_metrics import derive_business_output

    metrics.workload_phase = phase
    stats = get_gpu_stats()
    mem_pct = round(metrics.peak_memory_mb / total_vram * 100, 1) if total_vram > 0 else 0
    fits = metrics.peak_memory_mb < total_vram * 0.95 if total_vram > 0 else True

    metrics.operational_condition = {
        "num_users": num_users,
        "context_length": "n/a",
        "batch_size": 1,
        "fits_in_memory": fits,
        "gpu_util_pct": stats.get("gpu_util_pct", "n/a"),
        "power_w": stats.get("power_w", "n/a"),
        "mem_pct": mem_pct,
    }
    metrics.business_output = derive_business_output(metrics, total_vram)
    metrics.primary_bottleneck = classify_bottleneck(metrics, total_vram)
    # Flag spill explicitly
    if not fits:
        metrics.primary_bottleneck = "Memory Capacity"


class MemoryStress:
    """Stress test GPU memory utilisation with configurable load levels."""

    @staticmethod
    def get_available_memory_mb() -> float:
        if not torch.cuda.is_available():
            return 0
        torch.cuda.empty_cache()
        return torch.cuda.mem_get_info()[0] / 1024 / 1024

    @staticmethod
    def get_used_memory_mb() -> float:
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated() / 1024 / 1024

    @staticmethod
    def fill_memory_benchmark(
        target_percent: float = 0.85, precision: str = "FP32", num_runs: int = 3
    ) -> BenchmarkMetrics:
        """Fill GPU memory to target % and benchmark sustained bandwidth."""
        metrics = BenchmarkMetrics("Memory Fill", precision)
        total_vram = get_total_vram_mb()
        tensor = result = None  # predeclare so finally can always release them

        try:
            if not torch.cuda.is_available():
                metrics.error = "No CUDA device"
                return metrics

            dtype_map = {
                "FP32": (torch.float32, 4), "FP16": (torch.float16, 2),
                "BF16": (torch.bfloat16, 2), "FP64": (torch.float64, 8),
            }
            if precision not in dtype_map:
                metrics.error = f"Unsupported precision: {precision}"
                return metrics
            dtype, element_size = dtype_map[precision]

            total_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            headroom_mb = total_memory_mb * 0.10
            target_memory_mb = (total_memory_mb - headroom_mb) * target_percent
            num_elements = int((target_memory_mb * 1024 * 1024) / element_size)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            tensor = torch.randn(num_elements, dtype=dtype, device="cuda")
            torch.cuda.synchronize()
            initial_memory_mb = MemoryStress.get_used_memory_mb()

            latencies = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = tensor * 2.0
                result = result + 1.0
                result = result.sqrt()
                result = result * tensor
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            peak_memory_mb = get_gpu_peak_memory_mb()
            bytes_accessed = num_elements * element_size * 8
            bandwidth_gbs = (bytes_accessed / 1e9) / (avg_latency / 1000)

            metrics.latency_ms = avg_latency
            metrics.peak_memory_mb = peak_memory_mb
            metrics.memory_mb = initial_memory_mb
            metrics.throughput_samples_per_sec = bandwidth_gbs
            metrics._raw_latencies = latencies

            _tag_memory(metrics, "memory-fill", 1, total_vram)
            metrics.business_output["bandwidth_gbs"] = round(bandwidth_gbs, 1)
            metrics.business_output["fill_pct"] = round(target_percent * 100)

            return metrics

        except Exception as e:
            metrics.error = str(e)
            return metrics
        finally:
            # This benchmark deliberately fills VRAM — always release it, even on OOM.
            tensor = result = None
            free_cuda_memory()

    @staticmethod
    def concurrent_models_benchmark(
        num_models: int = 2,
        model_size_gb: float = 10.0,
        precision: str = "FP32",
    ) -> BenchmarkMetrics:
        """
        Simulate N concurrent user sessions by loading N model tensors and
        running them sequentially. Reports whether they fit, spill detection,
        and per-model throughput.
        """
        metrics = BenchmarkMetrics(f"{num_models} Concurrent Models", precision)
        total_vram = get_total_vram_mb()
        models = concurrent_results = None  # predeclare for finally cleanup

        try:
            if not torch.cuda.is_available():
                metrics.error = "No CUDA device"
                return metrics

            dtype_map = {
                "FP32": (torch.float32, 4), "FP16": (torch.float16, 2),
                "BF16": (torch.bfloat16, 2), "FP64": (torch.float64, 8),
            }
            if precision not in dtype_map:
                metrics.error = f"Unsupported precision: {precision}"
                return metrics
            dtype, element_size = dtype_map[precision]

            model_bytes = model_size_gb * 1024 * 1024 * 1024
            elements_per_model = int(model_bytes / element_size)
            required_vram_mb = num_models * model_size_gb * 1024

            # Detect spill before attempting load
            spills = total_vram > 0 and required_vram_mb > total_vram * 0.95

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            models = []
            loaded = 0
            for i in range(num_models):
                try:
                    model_tensor = torch.randn(elements_per_model, dtype=dtype, device="cuda")
                    models.append(model_tensor)
                    loaded += 1
                except RuntimeError:
                    # OOM — record partial load
                    spills = True
                    break

            torch.cuda.synchronize()
            memory_loaded_mb = MemoryStress.get_used_memory_mb()

            latencies = []
            for _ in range(3):
                if torch.cuda.is_available():
                    # Each session gets its own CUDA stream — true concurrent GPU execution
                    streams = [torch.cuda.Stream() for _ in range(loaded)]
                    concurrent_results = []
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for model_t, stream in zip(models, streams):
                        with torch.cuda.stream(stream):
                            concurrent_results.append(model_t * 2.0 + 1.0)
                    for s in streams:
                        s.synchronize()
                    latencies.append((time.perf_counter() - start) * 1000)
                else:
                    start = time.perf_counter()
                    concurrent_results = [m * 2.0 + 1.0 for m in models]
                    latencies.append((time.perf_counter() - start) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            peak_memory_mb = get_gpu_peak_memory_mb()

            metrics.latency_ms = avg_latency
            metrics.peak_memory_mb = peak_memory_mb
            metrics.memory_mb = memory_loaded_mb
            metrics.throughput_samples_per_sec = loaded / (avg_latency / 1000)
            metrics._raw_latencies = latencies

            _tag_memory(metrics, "multi-user-concurrency", num_models, total_vram)

            metrics.business_output["models_loaded"] = loaded
            metrics.business_output["models_requested"] = num_models
            metrics.business_output["spills"] = spills
            metrics.business_output["model_size_gb"] = model_size_gb
            metrics.business_output["required_vram_gb"] = round(required_vram_mb / 1024, 1)

            if spills:
                metrics.primary_bottleneck = "Memory Capacity"

            return metrics

        except Exception as e:
            metrics.error = str(e)
            return metrics
        finally:
            # Release all loaded model tensors and intermediate results, even on OOM.
            models = concurrent_results = None
            free_cuda_memory()
