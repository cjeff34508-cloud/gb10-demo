"""Vision model inference helper — with load timing and phase tagging."""

import gc
import json
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
from .mem_guard import fits_in_memory, GB10_USABLE_GB, GB10_RESERVE_GB
from .model_info import precision_compatible

_VLM_DIR = Path.home() / "gb10-demo" / "models" / "vlm-models"
_CNN_DIR = Path.home() / "gb10-demo" / "models" / "cnn-models"
# A true vision-LANGUAGE model may live with the LLMs (gb10-markets' Qwen3.6-35B
# is natively a VLM and is symlinked there), so the search includes that dir.
_LLM_DIR = Path.home() / "gb10-demo" / "models" / "llm-models"


def _local_path(model_name: str) -> str:
    slug = model_name.replace("/", "--")
    for base in (_VLM_DIR, _CNN_DIR, _LLM_DIR):
        local = base / slug
        if local.exists() and any(
            f.suffix in (".safetensors", ".bin", ".pt") for f in local.rglob("*") if f.is_file()
        ):
            return str(local)
    return model_name


# --------------------------------------------------------------------------- VLM detection
# "VLM" means two different things in this suite and conflating them is why the
# vision track was misleading. CLIP / ViT / DINO are vision ENCODERS: they turn an
# image into a vector and cannot answer a question about it. A vision-LANGUAGE
# model takes image + prompt and writes text. Customers mean the second one.
_VLM_ARCH_MARKERS = ("conditionalgeneration", "imagetexttotext", "visionencoderdecoder",
                     "forvision2seq", "omni", "vl")


def is_vision_language_model(model_path: str) -> bool:
    """True when the checkpoint can do image+prompt -> text, not just encode."""
    cfg = Path(model_path) / "config.json"
    if not cfg.exists():
        return False
    try:
        c = json.loads(cfg.read_text())
    except Exception:
        return False
    archs = " ".join(c.get("architectures") or []).lower()
    has_vision = bool(c.get("vision_config") or (c.get("text_config") or {}).get("vision_config"))
    return has_vision and any(m in archs for m in _VLM_ARCH_MARKERS)


class VisionModelInference:
    """Vision model inference wrapper with phase tracking."""

    def __init__(self, model_name: str, precision: str = "FP32"):
        self.model_name = model_name
        self.precision = precision
        self.model = None
        self.processor = None
        self._load_ms: float = 0.0
        self.is_vlm: bool = False          # image+prompt -> text, not just encode
        self.load_block_reason: str = ""
        self.incompatible: bool = False
        self.incompatible_reason: str = ""

    def load_model(self) -> bool:
        self.unload()
        try:
            print(f"Loading {self.model_name} in {self.precision}...")
            model_path = _local_path(self.model_name)
            self.is_vlm = is_vision_language_model(model_path)

            # --- guards the vision path previously had NONE of -----------------
            # CLIP/ViT/DINO are a few hundred MB, so this never mattered. A 120B
            # VLM is 75 GB and would have been loaded with no check at all on a
            # box whose one catastrophic failure was a memory over-commit.
            _compat, _why = precision_compatible(self.model_name, self.precision)
            if not _compat:
                self.incompatible = True
                self.incompatible_reason = _why
                self.load_block_reason = _why
                print(f"⛔ {self.precision} not compatible with {self.model_name}: {_why}")
                return False

            _ok, _est = fits_in_memory(self.model_name, self.precision)
            if not _ok:
                self.load_block_reason = (
                    f"~{_est:.0f} GB estimated > {GB10_USABLE_GB:.0f} GB usable "
                    f"({GB10_RESERVE_GB:.0f} GB reserved) — pick a lower precision "
                    f"or a smaller model"
                )
                print(f"✗ Memory guard blocked {self.model_name}: {self.load_block_reason}")
                return False
            if torch.cuda.is_available():
                _free = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                if _est * 1.10 > _free:
                    self.load_block_reason = (
                        f"~{_est:.0f} GB needed but only {_free:.0f} GB free right now "
                        f"— free the GPU or choose a smaller model/precision"
                    )
                    print(f"✗ Memory guard blocked {self.model_name}: {self.load_block_reason}")
                    return False

            t0 = time.perf_counter()
            try:
                self.processor = AutoProcessor.from_pretrained(model_path)
            except Exception:
                self.processor = None  # not required; benchmark uses synthetic pixel_values

            device = "cuda" if torch.cuda.is_available() else "cpu"

            if self.is_vlm:
                # A vision-LANGUAGE model needs the generative Auto class; AutoModel
                # would give back the bare backbone with no lm_head and no
                # .generate(), which is exactly why this track could only ever
                # encode. dtype is left to the checkpoint for quantized weights
                # (fp8/compressed-tensors carry their own), cast otherwise.
                from transformers import AutoModelForImageTextToText
                _kw = dict(device_map=device)
                if self.precision in ("FP16", "BF16"):
                    _kw["dtype"] = torch.float16 if self.precision == "FP16" else torch.bfloat16
                self.model = AutoModelForImageTextToText.from_pretrained(model_path, **_kw)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._load_ms = (time.perf_counter() - t0) * 1000
                self.model.eval()
                print(f"✓ VLM loaded in {self._load_ms:.0f} ms (image+prompt → text)")
                return True

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

    # ---------------------------------------------------------------- VLM path
    def benchmark_vlm(self, image, prompt: str, num_runs: int = 1,
                      max_new_tokens: int = 96) -> BenchmarkMetrics:
        """Benchmark image+prompt -> TEXT, and return what the model actually said.

        This is the measurement the vision track was missing. The encoder path
        below reports images/sec on SYNTHETIC random pixels — a number that says
        nothing about whether the model understood anything. Here the output text
        is part of the result, so a chart-reading claim can be checked against
        the figures the chart was drawn from.
        """
        metrics = BenchmarkMetrics(self.model_name, self.precision)
        if self.model is None:
            metrics.error = "Model not loaded"
            return metrics
        if not self.is_vlm:
            metrics.error = ("not a vision-language model — this checkpoint can only "
                             "encode images, not answer questions about them")
            return metrics
        if self.processor is None:
            metrics.error = "no AutoProcessor for this checkpoint; cannot build image+text input"
            return metrics
        try:
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            try:
                chat = self.processor.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False)
            except Exception:
                chat = prompt      # processors without a chat template
            inputs = self.processor(images=image, text=chat, return_tensors="pt")
            inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}

            with torch.no_grad():                       # warm-up: excludes compile/alloc
                self.model.generate(**inputs, max_new_tokens=4, do_sample=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            lat, texts, n_new = [], [], 0
            with torch.no_grad():
                for _ in range(max(1, num_runs)):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                              do_sample=False)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    lat.append((time.perf_counter() - t0) * 1000)
                    in_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
                    gen = out[0][in_len:]
                    n_new = int(gen.shape[-1])
                    texts.append(self.processor.decode(gen, skip_special_tokens=True).strip())

            avg = sum(lat) / len(lat)
            peak = get_gpu_peak_memory_mb()
            total = get_total_vram_mb()
            tps = (n_new * 1000.0 / avg) if avg > 0 else 0.0

            metrics.latency_ms = avg
            metrics.peak_memory_mb = peak
            metrics.memory_mb = peak
            metrics.tokens_per_sec = tps
            metrics.throughput_samples_per_sec = (1000.0 / avg) if avg > 0 else 0.0
            metrics._raw_latencies = lat
            metrics.workload_phase = "encode+decode"

            from .bottleneck_analyzer import get_gpu_stats
            st = get_gpu_stats()
            metrics.operational_condition = {
                "num_users": 1, "context_length": "image+prompt",
                "batch_size": 1,
                "fits_in_memory": peak < total * 0.95 if total > 0 else True,
                "gpu_util_pct": st.get("gpu_util_pct", "n/a"),
                "power_w": st.get("power_w", "n/a"),
                "mem_pct": round(peak / total * 100, 1) if total > 0 else 0,
            }
            from .business_metrics import derive_business_output
            metrics.business_output = derive_business_output(metrics, total)
            metrics.business_output.update({
                "load_ms": round(self._load_ms, 2),
                "ttft_ms": round(avg, 2),
                "tokens_per_sec": round(tps, 1),
                "new_tokens": n_new,
                "answer": texts[-1] if texts else "",
                "prompt": prompt,
            })
        except Exception as e:
            metrics.error = f"{type(e).__name__}: {e}"
        return metrics

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
