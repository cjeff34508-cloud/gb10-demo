"""HPC compute benchmarking helper — with workload-phase and bottleneck tagging."""

from typing import Tuple
import time

import torch
import numpy as np

from .benchmark_utils import (
    BenchmarkMetrics, get_gpu_peak_memory_mb, get_total_vram_mb, free_cuda_memory,
)


def _tag_hpc(metrics: BenchmarkMetrics, phase: str, total_vram: float):
    """Attach all 4 dimensions to an HPC BenchmarkMetrics result."""
    from .bottleneck_analyzer import classify_bottleneck, get_gpu_stats
    from .business_metrics import derive_business_output

    metrics.workload_phase = phase

    stats = get_gpu_stats()
    mem_pct = round(metrics.peak_memory_mb / total_vram * 100, 1) if total_vram > 0 else 0
    metrics.operational_condition = {
        "num_users": 1,
        "context_length": "n/a",
        "batch_size": 1,
        "fits_in_memory": metrics.peak_memory_mb < total_vram * 0.95 if total_vram > 0 else True,
        "gpu_util_pct": stats.get("gpu_util_pct", "n/a"),
        "power_w": stats.get("power_w", "n/a"),
        "mem_pct": mem_pct,
    }

    metrics.business_output = derive_business_output(metrics, total_vram)
    metrics.primary_bottleneck = classify_bottleneck(metrics, total_vram)


class HPCBenchmark:
    """HPC-style compute benchmarks (MatMul, bandwidth, etc.)."""

    @staticmethod
    def matmul_benchmark(
        matrix_size: int = 4096, precision: str = "FP32", num_runs: int = 3
    ) -> BenchmarkMetrics:
        """Benchmark matrix multiplication — compute-bound, shows peak TFLOPS."""
        metrics = BenchmarkMetrics("MatMul", precision)
        total_vram = get_total_vram_mb()

        # TF32 is a tensor-core matmul mode, not a dtype: FP32 (4-byte) storage
        # with reduced-precision accumulation. We force the global flag on for
        # TF32 and off for true FP32 so the two are measured distinctly.
        _prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            dtype_map = {"FP32": torch.float32, "TF32": torch.float32,
                         "FP16": torch.float16, "BF16": torch.bfloat16,
                         "FP64": torch.float64}
            if precision not in dtype_map:
                metrics.error = f"Unsupported precision: {precision}"
                return metrics
            dtype = dtype_map[precision]
            torch.backends.cuda.matmul.allow_tf32 = (precision == "TF32")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            a = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)
            b = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)

            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            for _ in range(1):
                _ = torch.matmul(a, b)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            latencies = []
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                c = torch.matmul(a, b)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            peak_memory = get_gpu_peak_memory_mb()
            flops = 2 * (matrix_size ** 3) / (avg_latency / 1000)

            metrics.latency_ms = avg_latency
            metrics.peak_memory_mb = peak_memory
            metrics.memory_mb = peak_memory
            metrics.throughput_samples_per_sec = flops / 1e12  # TFLOPs
            metrics._raw_latencies = latencies

            _tag_hpc(metrics, "compute", total_vram)
            metrics.business_output["tflops"] = round(flops / 1e12, 2)
            return metrics

        except Exception as e:
            metrics.error = str(e)
            return metrics
        finally:
            torch.backends.cuda.matmul.allow_tf32 = _prev_tf32
            free_cuda_memory()

    @staticmethod
    def bandwidth_benchmark(
        size_mb: int = 2048, precision: str = "FP32", num_runs: int = 3
    ) -> BenchmarkMetrics:
        """Benchmark GPU memory bandwidth — bandwidth-bound, shows GB/s vs peak."""
        metrics = BenchmarkMetrics("Bandwidth", precision)
        total_vram = get_total_vram_mb()

        try:
            dtype_map = {"FP32": (torch.float32, 4), "TF32": (torch.float32, 4), "FP16": (torch.float16, 2),
                         "BF16": (torch.bfloat16, 2), "FP64": (torch.float64, 8)}
            if precision not in dtype_map:
                metrics.error = f"Unsupported precision: {precision}"
                return metrics
            dtype, element_size = dtype_map[precision]

            num_elements = (size_mb * 1024 * 1024) // element_size
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tensor = torch.randn(num_elements, dtype=dtype, device=device)

            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

            latencies = []
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                temp = tensor * 2.0
                temp = temp + 1.0
                temp = temp.sqrt()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            peak_memory = get_gpu_peak_memory_mb()
            bytes_accessed = size_mb * 1024 * 1024 * 6  # 3 ops × read+write
            bandwidth_gbs = (bytes_accessed / 1e9) / (avg_latency / 1000)

            metrics.latency_ms = avg_latency
            metrics.peak_memory_mb = peak_memory
            metrics.memory_mb = peak_memory
            metrics.throughput_samples_per_sec = bandwidth_gbs
            metrics._raw_latencies = latencies

            _tag_hpc(metrics, "memory-bandwidth", total_vram)
            metrics.business_output["bandwidth_gbs"] = round(bandwidth_gbs, 1)
            return metrics

        except Exception as e:
            metrics.error = str(e)
            return metrics
        finally:
            free_cuda_memory()

    @staticmethod
    def reduction_benchmark(
        vector_size: int = 256_000_000, precision: str = "FP32", num_runs: int = 3
    ) -> BenchmarkMetrics:
        """Benchmark reduction ops on large vectors — shows memory-bandwidth utilisation."""
        metrics = BenchmarkMetrics("Reduction", precision)
        total_vram = get_total_vram_mb()

        try:
            dtype_map = {"FP32": torch.float32, "TF32": torch.float32,
                         "FP16": torch.float16, "BF16": torch.bfloat16,
                         "FP64": torch.float64}
            if precision not in dtype_map:
                metrics.error = f"Unsupported precision: {precision}"
                return metrics
            dtype = dtype_map[precision]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            vector = torch.randn(vector_size, dtype=dtype, device=device)
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

            latencies = []
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = vector.sum()
                _ = vector.max()
                _ = vector.mean()
                _ = vector.std()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            peak_memory = get_gpu_peak_memory_mb()

            metrics.latency_ms = avg_latency
            metrics.peak_memory_mb = peak_memory
            metrics.memory_mb = peak_memory
            metrics._raw_latencies = latencies

            _tag_hpc(metrics, "compute", total_vram)
            return metrics

        except Exception as e:
            metrics.error = str(e)
            return metrics
        finally:
            free_cuda_memory()

    @staticmethod
    def lob_benchmark(
        num_price_levels: int = 1_000_000,
        num_operations: int = 2_000_000,
        precision: str = "FP32",
        num_runs: int = 3,
    ) -> BenchmarkMetrics:
        """
        Simulate Limit Order Book random scatter/gather access patterns.

        Three internal phases — gather (price-level lookup), spread compute (BBO),
        scatter update (write-back) — tagged as 'lob-gather / lob-compute / lob-scatter'.
        Reports effective bandwidth (GB/s) so results compare directly to Bandwidth Test.
        """
        metrics = BenchmarkMetrics("LOB Bandwidth", precision)
        total_vram = get_total_vram_mb()

        try:
            dtype_map = {"FP32": (torch.float32, 4), "TF32": (torch.float32, 4), "FP16": (torch.float16, 2),
                         "BF16": (torch.bfloat16, 2), "FP64": (torch.float64, 8)}
            if precision not in dtype_map:
                metrics.error = f"Unsupported precision: {precision}"
                return metrics
            dtype, bytes_per_elem = dtype_map[precision]

            fields = 8
            entry_bytes = fields * bytes_per_elem
            device = "cuda" if torch.cuda.is_available() else "cpu"

            order_book = torch.randn(num_price_levels, fields, dtype=dtype, device=device)

            hot_top = max(10_000, num_price_levels // 100)
            n_hot = int(num_operations * 0.80)
            n_cold = num_operations - n_hot
            hot_idx = torch.randint(0, hot_top, (n_hot,), device=device)
            cold_idx = torch.randint(hot_top, num_price_levels, (n_cold,), device=device)
            random_indices = torch.cat([hot_idx, cold_idx])
            perm = torch.randperm(num_operations, device=device)
            random_indices = random_indices[perm]
            idx_expanded = random_indices.unsqueeze(1).expand(-1, fields)
            delta = torch.randn(num_operations, fields, dtype=dtype, device=device) * 0.0001

            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Warmup
            gathered = order_book[random_indices]
            _ = (gathered[:, 1] - gathered[:, 2]).abs().mean()
            order_book.scatter_add_(0, idx_expanded, delta)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latencies = []
            phase_gather, phase_compute, phase_scatter = [], [], []

            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                gathered = order_book[random_indices]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                spread = gathered[:, 1] - gathered[:, 2]
                _ = spread.abs().mean()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t2 = time.perf_counter()

                order_book.scatter_add_(0, idx_expanded, delta)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t3 = time.perf_counter()

                total_ms = (t3 - t0) * 1000
                latencies.append(total_ms)
                phase_gather.append((t1 - t0) * 1000)
                phase_compute.append((t2 - t1) * 1000)
                phase_scatter.append((t3 - t2) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            peak_memory = get_gpu_peak_memory_mb()
            bytes_accessed = 2 * num_operations * entry_bytes
            effective_bw_gbs = (bytes_accessed / 1e9) / (avg_latency / 1000)

            metrics.latency_ms = avg_latency
            metrics.peak_memory_mb = peak_memory
            metrics.memory_mb = peak_memory
            metrics.throughput_samples_per_sec = effective_bw_gbs
            metrics._raw_latencies = latencies

            _tag_hpc(metrics, "lob-gather+compute+scatter", total_vram)

            # Store per-phase breakdown in business_output for waterfall chart
            metrics.business_output["lob_gather_ms"] = round(sum(phase_gather) / len(phase_gather), 2)
            metrics.business_output["lob_compute_ms"] = round(sum(phase_compute) / len(phase_compute), 2)
            metrics.business_output["lob_scatter_ms"] = round(sum(phase_scatter) / len(phase_scatter), 2)
            metrics.business_output["bandwidth_gbs"] = round(effective_bw_gbs, 1)
            return metrics

        except Exception as e:
            metrics.error = str(e)
            return metrics
        finally:
            free_cuda_memory()


    @staticmethod
    def black_scholes_benchmark(
        num_options: int = 1_000_000, precision: str = "FP32", num_runs: int = 3
    ) -> BenchmarkMetrics:
        """
        Vectorized Black-Scholes option pricing on 1M contracts.
        Compute-bound + memory-bandwidth mix; typical quant finance workload.
        Reports M options/sec and estimated GFLOPS.
        """
        metrics = BenchmarkMetrics("Black-Scholes", precision)
        total_vram = get_total_vram_mb()

        try:
            dtype_map = {
                "FP32": (torch.float32, 4), "TF32": (torch.float32, 4), "FP16": (torch.float16, 2),
                "BF16": (torch.bfloat16, 2), "FP64": (torch.float64, 8),
            }
            if precision not in dtype_map:
                metrics.error = f"Unsupported precision: {precision}"
                return metrics
            dtype, _ = dtype_map[precision]
            device = "cuda" if torch.cuda.is_available() else "cpu"

            S     = torch.rand(num_options, dtype=dtype, device=device) * 150 + 50
            K     = torch.rand(num_options, dtype=dtype, device=device) * 150 + 50
            T     = torch.rand(num_options, dtype=dtype, device=device) + 0.1
            r     = torch.full((num_options,), 0.05, dtype=dtype, device=device)
            sigma = torch.rand(num_options, dtype=dtype, device=device) * 0.3 + 0.1

            # 1/sqrt(2) as a Python float — torch broadcasts it correctly against any
            # tensor dtype (FP16/BF16/FP32/FP64). (A torch.dtype is not callable, so the
            # old dtype(...) form raised TypeError on FP64.)
            _SQRT2_INV = 0.7071067811865476

            def _compute():
                sqrt_T = torch.sqrt(T)
                d1 = (torch.log(S / K) + (r + 0.5 * sigma.pow(2)) * T) / (sigma * sqrt_T)
                d2 = d1 - sigma * sqrt_T
                N_d1 = 0.5 * (1.0 + torch.erf(d1 * _SQRT2_INV))
                N_d2 = 0.5 * (1.0 + torch.erf(d2 * _SQRT2_INV))
                call = S * N_d1 - K * torch.exp(-r * T) * N_d2
                put  = call - S + K * torch.exp(-r * T)
                return call, put

            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            with torch.no_grad():
                _compute()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latencies = []
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _compute()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            peak_memory = get_gpu_peak_memory_mb()
            options_per_sec = num_options / (avg_latency / 1000)
            gflops = options_per_sec * 20 / 1e9  # ~20 FLOPs/option

            metrics.latency_ms = avg_latency
            metrics.peak_memory_mb = peak_memory
            metrics.memory_mb = peak_memory
            metrics.throughput_samples_per_sec = options_per_sec
            metrics._raw_latencies = latencies

            _tag_hpc(metrics, "compute+memory-bandwidth", total_vram)
            metrics.business_output["options_per_sec_M"] = round(options_per_sec / 1e6, 2)
            metrics.business_output["gflops"] = round(gflops, 1)
            metrics.business_output["num_options"] = num_options
            return metrics

        except Exception as e:
            metrics.error = str(e)
            return metrics
        finally:
            free_cuda_memory()

    @staticmethod
    def montecarlo_var_benchmark(
        num_paths: int = 1_000_000,
        num_assets: int = 100,
        horizon_days: int = 10,
        confidence: float = 0.99,
        precision: str = "FP32",
        num_runs: int = 3,
    ) -> BenchmarkMetrics:
        """
        Monte Carlo Value-at-Risk: simulate 1M GBM portfolio paths over a 10-day horizon,
        compute 99th-percentile loss. Representative risk management workload.
        Reports M paths/sec and the VaR estimate.
        """
        metrics = BenchmarkMetrics("Monte Carlo VaR", precision)
        total_vram = get_total_vram_mb()

        try:
            dtype_map = {
                "FP32": (torch.float32, 4), "TF32": (torch.float32, 4), "FP16": (torch.float16, 2),
                "BF16": (torch.bfloat16, 2), "FP64": (torch.float64, 8),
            }
            if precision not in dtype_map:
                metrics.error = f"Unsupported precision: {precision}"
                return metrics
            dtype, _ = dtype_map[precision]
            device = "cuda" if torch.cuda.is_available() else "cpu"

            mu_daily    = torch.rand(num_assets, dtype=dtype, device=device) * 0.10 / 252
            sigma_daily = (torch.rand(num_assets, dtype=dtype, device=device) * 0.20 + 0.10) / (252 ** 0.5)

            def _compute():
                # [num_paths, num_assets, horizon_days] random shocks
                Z = torch.randn(num_paths, num_assets, horizon_days, dtype=dtype, device=device)
                log_ret = (
                    mu_daily.view(1, -1, 1)
                    - 0.5 * sigma_daily.view(1, -1, 1).pow(2)
                    + sigma_daily.view(1, -1, 1) * Z
                )
                # Equal-weight portfolio cumulative log-return
                port_return = log_ret.sum(dim=2).mean(dim=1)
                # torch.quantile only accepts float32/float64 — cast up from FP16/BF16.
                losses = (-port_return).to(torch.float32) if port_return.dtype in (
                    torch.float16, torch.bfloat16) else -port_return
                var_val = torch.quantile(losses, confidence)
                return var_val

            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            with torch.no_grad():
                _compute()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latencies, var_vals = [], []
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    v = _compute()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)
                var_vals.append(float(v.item()))

            avg_latency = sum(latencies) / len(latencies)
            peak_memory = get_gpu_peak_memory_mb()
            paths_per_sec = num_paths / (avg_latency / 1000)

            metrics.latency_ms = avg_latency
            metrics.peak_memory_mb = peak_memory
            metrics.memory_mb = peak_memory
            metrics.throughput_samples_per_sec = paths_per_sec
            metrics._raw_latencies = latencies

            _tag_hpc(metrics, "monte-carlo-var", total_vram)
            metrics.business_output["paths_per_sec_M"] = round(paths_per_sec / 1e6, 2)
            metrics.business_output["var_99_pct"] = round((sum(var_vals) / len(var_vals)) * 100, 3)
            metrics.business_output["num_paths"] = num_paths
            metrics.business_output["horizon_days"] = horizon_days
            return metrics

        except Exception as e:
            metrics.error = str(e)
            return metrics
        finally:
            free_cuda_memory()


def benchmark_hpc(precisions: list = None) -> dict:
    if precisions is None:
        precisions = ["FP32", "FP64"]
    results = {"matmul": [], "bandwidth": [], "reduction": []}
    for precision in precisions:
        results["matmul"].append(HPCBenchmark.matmul_benchmark(precision=precision))
        results["bandwidth"].append(HPCBenchmark.bandwidth_benchmark(precision=precision))
        results["reduction"].append(HPCBenchmark.reduction_benchmark(precision=precision))
    return results
