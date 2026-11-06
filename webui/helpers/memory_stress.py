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
    def dual_model_serving_benchmark(
        precision: str = "BF16", models: list | None = None
    ) -> BenchmarkMetrics:
        """Two DIFFERENT real models co-resident, each asked to generate.

        Replaces a synthetic version that allocated two 10 GB tensors of
        random noise and multiplied them in place. That measured arithmetic on
        random numbers; this measures the thing the name claims — whether the
        box can hold two models at once and serve from both, which is the real
        enterprise pattern (e.g. a chat model beside a code model).

        Uses the two SMALLEST models present, because the question is co-residency
        and footprint, not which model is fastest.
        """
        from .llm_inference import LLMInference, _local_path, smallest_local_model
        from .mem_guard import params_b

        metrics = BenchmarkMetrics("Dual Model Serving", precision)
        total_vram = get_total_vram_mb()

        if models is None:
            import json
            from pathlib import Path as _P
            try:
                cfg = json.loads(
                    (_P.home() / "gb10-demo" / "config" / "model-lists.json").read_text())
                present = [m for m in (cfg.get("llm") or []) if _local_path(m) != m]
            except Exception:
                present = []
            # two smallest DISTINCT checkpoints
            models = sorted(set(present), key=params_b)[:2]

        if len(models) < 2:
            metrics.error = ("Needs two models on disk; found "
                             f"{len(models)}. Download another via scripts/model-downloader.py")
            return metrics

        loaded, infs, per_model = 0, [], []
        try:
            for name in models:
                inf = LLMInference(name, precision, max_tokens=32)
                if not inf.load_model():
                    metrics.error = (f"{name} failed to load at {precision}: "
                                     f"{getattr(inf, 'load_block_reason', '') or 'see log'}")
                    break
                infs.append(inf)
                loaded += 1

            if loaded == 2:
                t0 = time.perf_counter()
                for inf in infs:
                    txt = inf.generate_text("Summarise in one sentence why "
                                            "memory bandwidth limits decode speed.",
                                            max_new_tokens=32)
                    per_model.append({
                        "model": inf.model_name,
                        "params_b": params_b(inf.model_name),
                        "chars": len(txt or ""),
                        "coherent": bool(txt and sum(c.isascii() for c in txt) / max(len(txt), 1) > 0.9),
                    })
                metrics.latency_ms = (time.perf_counter() - t0) * 1000

            metrics.peak_memory_mb = get_gpu_peak_memory_mb()
            metrics.memory_mb = metrics.peak_memory_mb
            if metrics.latency_ms > 0:
                metrics.throughput_samples_per_sec = loaded / (metrics.latency_ms / 1000)

            _tag_memory(metrics, "dual-model-serving", loaded, total_vram)
            metrics.business_output["models_loaded"] = loaded
            metrics.business_output["models_requested"] = 2
            metrics.business_output["model_names"] = list(models)
            metrics.business_output["per_model"] = per_model
            metrics.business_output["spills"] = loaded < 2
            metrics.business_output["combined_gb"] = round(metrics.peak_memory_mb / 1024, 2)
            if loaded < 2:
                metrics.primary_bottleneck = "Memory Capacity"
            return metrics

        except Exception as e:
            metrics.error = str(e)
            return metrics
        finally:
            for inf in infs:
                try:
                    inf.unload()
                except Exception:
                    pass
            free_cuda_memory()

    # NOTE: concurrent_models_benchmark() was removed 2026-11-06. It allocated N
    # tensors of torch.randn() noise and ran mul_().add_() across CUDA streams,
    # reporting "samples/sec" for arithmetic on random numbers while the UI
    # presented it as multi-user serving; the confidence panel already flagged it
    # SYNTHETIC in red. Multi-User Concurrency now batches N real sequences
    # through ONE real model copy (helpers/llm_inference.LLMInference.benchmark
    # with batch_size=N), which is what continuous batching does and which
    # reports the same per-user/aggregate pair the vLLM captures record. Dual
    # Model Serving uses dual_model_serving_benchmark() above. Do not reinstate a
    # synthetic stand-in for a measurement.
