"""Vision model inference helper — with load timing and phase tagging."""

import gc
import time
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoModel, AutoProcessor

from .benchmark_utils import (
    BenchmarkMetrics,
    get_gpu_peak_memory_mb,
    get_total_vram_mb,
    quantize_to_precision,
)

_VLM_DIR = Path.home() / "gb10-demo" / "models" / "vlm-models"
_CNN_DIR = Path.home() / "gb10-demo" / "models" / "cnn-models"


def _local_path(model_name: str) -> str:
    slug = model_name.replace("/", "--")
    for base in (_VLM_DIR, _CNN_DIR):
        local = base / slug
        if local.exists() and any(
            f.suffix in (".safetensors", ".bin", ".pt") for f in local.rglob("*") if f.is_file()
        ):
            return str(local)
    return model_name


class VisionModelInference:
    """Vision model inference wrapper with phase tracking."""

    def __init__(self, model_name: str, precision: str = "FP32"):
        self.model_name = model_name
        self.precision = precision
        self.model = None
        self.processor = None
        self._load_ms: float = 0.0

    def load_model(self) -> bool:
        self.unload()
        try:
            print(f"Loading {self.model_name} in {self.precision}...")
            model_path = _local_path(self.model_name)

            t0 = time.perf_counter()
            try:
                self.processor = AutoProcessor.from_pretrained(model_path)
            except Exception:
                self.processor = None  # not required; benchmark uses synthetic pixel_values

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.precision == "FP16":
                self.model = AutoModel.from_pretrained(
                    model_path, torch_dtype=torch.float16, device_map=device
                )
            elif self.precision == "BF16":
                self.model = AutoModel.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device_map=device
                )
            else:  # FP32, INT8, FP4, NVFP4
                self.model = AutoModel.from_pretrained(model_path, device_map=device)

            if self.precision in ("INT8", "FP4", "NVFP4"):
                self._apply_quantization()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._load_ms = (time.perf_counter() - t0) * 1000

            self.model.eval()
            print(f"✓ Model loaded in {self._load_ms:.0f} ms")
            return True

        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False

    def _apply_quantization(self):
        with torch.no_grad():
            for param in self.model.parameters():
                param.data = quantize_to_precision(param.data, self.precision)

    def benchmark(self, num_runs: int = 3, batch_size: int = 1) -> BenchmarkMetrics:
        metrics = BenchmarkMetrics(self.model_name, self.precision)

        if self.model is None:
            metrics.error = "Model not loaded"
            return metrics

        try:
            device = next(self.model.parameters()).device
            pixel_values = torch.randn(
                batch_size, 3, 224, 224,
                dtype=next(self.model.parameters()).dtype,
                device=device,
            )

            self.model.eval()

            def _forward():
                if hasattr(self.model, "get_image_features"):
                    return self.model.get_image_features(pixel_values=pixel_values)
                return self.model(pixel_values=pixel_values)

            # Warmup
            with torch.no_grad():
                _forward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

            latencies = []
            with torch.no_grad():
                for _ in range(num_runs):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    _forward()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - start) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            peak_memory = get_gpu_peak_memory_mb()
            total_vram = get_total_vram_mb()
            images_per_sec = (batch_size * 1000) / avg_latency if avg_latency > 0 else 0

            metrics.latency_ms = avg_latency
            metrics.peak_memory_mb = peak_memory
            metrics.memory_mb = peak_memory
            metrics.throughput_samples_per_sec = images_per_sec
            metrics.tokens_per_sec = images_per_sec
            metrics._raw_latencies = latencies
            metrics.workload_phase = "encode"

            # Dimension 3: Operating condition
            from .bottleneck_analyzer import get_gpu_stats
            stats = get_gpu_stats()
            mem_pct = round(peak_memory / total_vram * 100, 1) if total_vram > 0 else 0
            metrics.operational_condition = {
                "num_users": 1,
                "context_length": "n/a",
                "batch_size": batch_size,
                "fits_in_memory": peak_memory < total_vram * 0.95 if total_vram > 0 else True,
                "gpu_util_pct": stats.get("gpu_util_pct", "n/a"),
                "power_w": stats.get("power_w", "n/a"),
                "mem_pct": mem_pct,
            }

            # Dimension 4: Business output
            from .business_metrics import derive_business_output
            metrics.business_output = derive_business_output(metrics, total_vram)
            metrics.business_output["load_ms"] = round(self._load_ms, 2)
            metrics.business_output["images_per_sec"] = round(images_per_sec, 1)
            # Vision has no prefill/decode split — encode is the only phase
            metrics.business_output["ttft_ms"] = round(avg_latency, 2)
            metrics.business_output["decode_ms"] = 0.0

            # Dimension 2: Bottleneck
            from .bottleneck_analyzer import classify_bottleneck
            metrics.primary_bottleneck = classify_bottleneck(metrics, total_vram)

            return metrics

        except Exception as e:
            metrics.error = str(e)
            return metrics

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def benchmark_vision_model(
    model_name: str, precisions: list = None, batch_size: int = 1
) -> list:
    if precisions is None:
        precisions = ["FP32", "FP16"]
    results = []
    for precision in precisions:
        inference = VisionModelInference(model_name, precision)
        if inference.load_model():
            metrics = inference.benchmark(batch_size=batch_size)
            results.append(metrics)
        else:
            m = BenchmarkMetrics(model_name, precision)
            m.error = "Failed to load model"
            results.append(m)
        inference.unload()
    return results
