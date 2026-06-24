"""LLM inference helper — phase-separated benchmarking (load / prefill / decode)."""

import gc
import time
import threading
from pathlib import Path
from typing import Optional
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer,
)

from .benchmark_utils import BenchmarkMetrics, get_gpu_peak_memory_mb, get_total_vram_mb
from .mem_guard import fits_in_memory, estimate_model_gb, GB10_USABLE_GB, GB10_RESERVE_GB
from .model_info import precision_compatible

_MODELS_BASE = Path.home() / "gb10-demo" / "models"
_MODELS_DIR  = _MODELS_BASE / "llm-models"
# Search order for local checkpoints: benchmark LLMs, then the always-on
# narrator dir (Qwen3.5-35B-A3B lives in narrator-models/).
_MODEL_DIRS  = [_MODELS_DIR, _MODELS_BASE / "narrator-models"]

# Prompt long enough to produce a meaningful prefill measurement
_DEFAULT_PROMPT = (
    "Explain the key advantages of running AI inference locally on dedicated hardware "
    "versus cloud-hosted GPU instances for enterprise workloads:"
)


def _local_path(model_name: str) -> str:
    slug = model_name.replace("/", "--")
    for base in _MODEL_DIRS:
        local = base / slug
        if local.exists() and any(
            f.suffix in (".safetensors", ".bin", ".pt") for f in local.rglob("*") if f.is_file()
        ):
            return str(local)
    return model_name


def _warm_file_cache(model_path: str) -> None:
    """Read the checkpoint's weight files into the OS page cache.

    Per-precision load timings are otherwise order-dependent: the precision that
    runs first pays the cold-NVMe read of the checkpoint while later precisions
    read the same files from RAM, making them look artificially fast. Warming the
    cache here — *before* the load timer starts — puts every precision on equal
    footing (all warm), so the measured load time reflects real work
    (quantization vs plain cast, data volume) rather than disk-cache luck.

    Best-effort: ignores non-local paths (HF repo ids) and IO errors.
    """
    p = Path(model_path)
    if not p.is_dir():
        return
    for f in sorted(p.rglob("*")):
        if f.suffix in (".safetensors", ".bin", ".pt") and f.is_file():
            try:
                with open(f, "rb", buffering=0) as fh:
                    while fh.read(16 * 1024 * 1024):
                        pass
            except OSError:
                pass


def _is_prequantized_nvfp4(model_name: str) -> bool:
    """True for NVIDIA ModelOpt pre-quantized FP4 checkpoints."""
    return "NVFP4" in model_name.upper() or "NVF4" in model_name.upper()


def _estimate_params_b(model_name: str) -> float:
    """Active parameter count (billions) for the effective-TFLOPS estimate.

    NOTE: this is intentionally distinct from mem_guard.params_b, which reports
    *total* weights for the memory estimate. For MoE models the two differ on
    purpose — Mixtral-8x7B activates ~12.9B params per token (compute) but loads
    all 46.7B (memory). Keep them separate; do not "reconcile" the values.
    """
    n = model_name.lower()
    for key, val in [
        ("mixtral", 12.9), ("phi-4", 14.0), ("phi-3", 3.8),
        ("1.1b", 1.1), ("1.5b", 1.5), ("3.8b", 3.8),
        ("72b", 72.0), ("70b", 70.0), ("32b", 32.0),
        ("14b", 14.0), ("13b", 13.0), ("8b", 8.0),
        ("7b", 7.0), ("3b", 3.0), ("1b", 1.0),
    ]:
        if key in n:
            return val
    return 7.0


class LLMInference:
    """LLM inference wrapper with load / prefill / decode phase tracking."""

    def __init__(self, model_name: str, precision: str = "FP32", max_tokens: int = 128):
        self.model_name = model_name
        self.precision = precision
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None
        self._load_ms: float = 0.0
        self.incompatible: bool = False   # True if precision can't run on this checkpoint
        self.incompatible_reason: str = ""

    def load_model(self) -> bool:
        # Purge any previously loaded model before allocating new GPU memory.
        self.unload()
        try:
            print(f"Loading {self.model_name} in {self.precision}...")
            model_path = _local_path(self.model_name)

            # Memory guard: refuse to load if estimated footprint exceeds usable GB10 memory.
            _ok, _est_gb = fits_in_memory(self.model_name, self.precision)
            if not _ok:
                print(
                    f"✗ Memory guard blocked {self.model_name} @ {self.precision}: "
                    f"~{_est_gb:.0f} GB estimated > {GB10_USABLE_GB:.0f} GB usable "
                    f"({GB10_RESERVE_GB:.0f} GB reserved for OS/runtime). "
                    f"Choose a lower-precision or smaller model."
                )
                return False

            # Precision incompatibility (e.g. pre-quantized NVFP4 checkpoints, whose
            # packed FP4 weights only load at FP4/NVFP4). Report it as Not Compatible
            # rather than silently switching precision and running.
            _compat, _why = precision_compatible(self.model_name, self.precision)
            if not _compat:
                self.incompatible = True
                self.incompatible_reason = _why
                print(f"⛔ {self.precision} not compatible with {self.model_name}: {_why}")
                return False

            # Warm the OS page cache first so the timed load below starts from the
            # same cache state for every precision (otherwise whichever precision
            # runs first eats the cold-disk read and later ones look artificially
            # fast). Makes per-precision load timing a fair comparison.
            _warm_file_cache(model_path)

            t0 = time.perf_counter()

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # On GB10/GH200 unified memory, nvidia-smi/cuda mem_get_info is
            # unreliable, so device_map="auto" under-counts free VRAM once the
            # GPU holds any prior allocation (e.g. right after a benchmark) and
            # spuriously offloads layers to CPU — which makes bitsandbytes refuse
            # to load. Pin the GPU budget so placement stays fully on-device.
            _max_memory = {0: f"{int(GB10_USABLE_GB)}GiB"}

            if self.precision == "INT8":
                # Quantization stays available — but if it genuinely can't load,
                # report failure; never silently fall back to FP32 and run.
                try:
                    qcfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path, quantization_config=qcfg,
                        device_map="auto", max_memory=_max_memory,
                    )
                    print("✓ Using bitsandbytes INT8 quantization")
                except Exception as e:
                    print(f"✗ INT8 quantization unavailable for {self.model_name}: {e}")
                    return False

            elif self.precision in ("FP4", "NVFP4"):
                if _is_prequantized_nvfp4(self.model_name):
                    # Pre-quantized FP4 checkpoint — weights are already packed
                    # 4-bit, so loading skips the huge BF16 staging that OOMs the
                    # GB10. NOTE: do NOT pass ignore_mismatched_sizes=True — that
                    # silently re-inits mismatched layers with random weights and
                    # reports false success. Let a bad checkpoint fail loudly.
                    #
                    # compressed-tensors' setup path torch.compiles an FP4 cast
                    # kernel via Triton, which needs python3-dev headers (absent
                    # on this box). For the run_compressed=False (BF16-decompress)
                    # load we don't need that kernel, so disable Dynamo to run the
                    # setup eagerly. (Real FP4 via run_compressed=True DOES need
                    # the kernel → install python3.12-dev and drop this guard.)
                    import torch._dynamo as _dyn
                    _dyn_prev = _dyn.config.disable
                    _dyn.config.disable = True
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.bfloat16,
                            device_map="auto",
                            max_memory=_max_memory,
                        )
                        print(f"✓ Pre-quantized FP4 checkpoint loaded")
                    except Exception as e:
                        print(f"✗ NVFP4 pre-quantized load failed: {e}")
                        return False
                    finally:
                        _dyn.config.disable = _dyn_prev
                else:
                    try:
                        qcfg = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_quant_type="fp4",
                            bnb_4bit_use_double_quant=True,
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path, quantization_config=qcfg,
                            device_map="auto", max_memory=_max_memory,
                        )
                        print(f"✓ Using bitsandbytes {self.precision} (4-bit) quantization")
                    except Exception as e:
                        # Quantization stays available — but never silently fall back
                        # to FP32 and run; report failure instead.
                        print(f"✗ {self.precision} quantization unavailable for {self.model_name}: {e}")
                        return False

            elif self.precision == "FP16":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float16,
                    device_map="auto", max_memory=_max_memory,
                )
            elif self.precision == "BF16":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16,
                    device_map="auto", max_memory=_max_memory,
                )
            else:  # FP32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map="auto", max_memory=_max_memory,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._load_ms = (time.perf_counter() - t0) * 1000

            self.model.eval()
            print(f"✓ Model loaded in {self._load_ms:.0f} ms")
            return True

        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False

    def benchmark(
        self,
        prompt: str = _DEFAULT_PROMPT,
        num_runs: int = 3,
        batch_size: int = 1,
        context_length: str = "short",
        num_users: int = 1,
    ) -> BenchmarkMetrics:
        """
        Run phased benchmark: prefill (TTFT) then decode.
        OOM is caught and returned as a Memory Capacity result rather than crashing.
        """
        metrics = BenchmarkMetrics(self.model_name, self.precision)

        if self.model is None or self.tokenizer is None:
            metrics.error = "Model not loaded"
            return metrics

        try:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # Adjust prompt length and output budget for context scenario
            if context_length == "long":
                full_prompt = (prompt + " ") * 6
                max_new = 256
                max_input_len = 1024
            else:
                full_prompt = prompt
                max_new = self.max_tokens
                max_input_len = 256

            prompts = [full_prompt] * batch_size
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_len,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            seq_len = inputs["input_ids"].shape[1]
            eos = self.tokenizer.eos_token_id

            # --- Phase 1: Prefill / TTFT ---
            # Warmup
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=eos)

            ttft_latencies = []
            for _ in range(max(2, num_runs)):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = self.model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=eos)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttft_latencies.append((time.perf_counter() - t0) * 1000)

            ttft_ms = sum(ttft_latencies) / len(ttft_latencies)

            # --- Phase 2: Full generation (prefill + decode) ---
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

            total_latencies = []
            for _ in range(max(2, num_runs)):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=False,
                        pad_token_id=eos,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_latencies.append((time.perf_counter() - t0) * 1000)

            avg_total_ms = sum(total_latencies) / len(total_latencies)
            decode_ms = max(0.0, avg_total_ms - ttft_ms)
            tokens_generated = max_new * batch_size
            tokens_per_sec = (tokens_generated * 1000) / decode_ms if decode_ms > 0 else 0.0
            peak_memory = get_gpu_peak_memory_mb()
            total_vram = get_total_vram_mb()

            metrics.latency_ms = avg_total_ms
            metrics.peak_memory_mb = peak_memory
            metrics.memory_mb = peak_memory
            metrics.tokens_per_sec = tokens_per_sec
            metrics.throughput_samples_per_sec = tokens_per_sec
            metrics._raw_latencies = total_latencies
            metrics.workload_phase = "prefill+decode"

            # --- Dimension 3: Operating condition ---
            from .bottleneck_analyzer import get_gpu_stats
            stats = get_gpu_stats()
            mem_pct = round(peak_memory / total_vram * 100, 1) if total_vram > 0 else 0
            metrics.operational_condition = {
                "num_users": num_users,
                "context_length": context_length,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "max_new_tokens": max_new,
                "fits_in_memory": peak_memory < total_vram * 0.95 if total_vram > 0 else True,
                "gpu_util_pct": stats.get("gpu_util_pct", "n/a"),
                "power_w": stats.get("power_w", "n/a"),
                "mem_pct": mem_pct,
            }

            # --- Dimension 4: Business output ---
            from .business_metrics import derive_business_output
            metrics.business_output = derive_business_output(metrics, total_vram)
            metrics.business_output["ttft_ms"] = round(ttft_ms, 2)
            metrics.business_output["decode_ms"] = round(decode_ms, 2)
            metrics.business_output["load_ms"] = round(self._load_ms, 2)
            metrics.business_output["tokens_per_sec"] = round(tokens_per_sec, 1)

            # --- Serving metrics ---
            _qps_val  = batch_size / (avg_total_ms / 1000) if avg_total_ms > 0 else 0
            _tpot_val = (decode_ms / max_new) if (max_new > 0 and decode_ms > 0) else 0
            # Memory bandwidth utilisation: model bytes read per second vs GB10's real
            # LPDDR5X unified-memory bandwidth (~273 GB/s). NVLink-C2C (900 GB/s) and the
            # "4 TB/s" interconnect figure are not the decode ceiling — LPDDR5X is.
            _model_gb  = peak_memory / 1024          # MB → GB
            _bw_gbs    = _model_gb * tokens_per_sec  # GB/s consumed during decode
            _bw_util   = min(100.0, _bw_gbs / 273 * 100) if tokens_per_sec > 0 else 0
            # Estimated effective TFLOPS (2 MACs × params × tokens / decode_time)
            _params_b  = _estimate_params_b(self.model_name)
            _tflops_est = (2 * _params_b * max_new * batch_size) / max(decode_ms / 1000, 1e-6) / 1e3
            _theo_tf   = {"FP32": 250, "FP16": 500, "BF16": 500, "INT8": 500,
                          "FP4": 1000, "NVFP4": 1000}.get(self.precision.split()[0], 500)
            _tfu       = min(100.0, _tflops_est / _theo_tf * 100)
            # P99 estimate from observed run spread
            _p99_est   = max(total_latencies) * 1.12 if total_latencies else 0

            metrics.business_output["qps"]             = round(_qps_val, 2)
            metrics.business_output["tpot_ms"]         = round(_tpot_val, 2)
            metrics.business_output["itl_ms"]          = round(_tpot_val, 2)
            metrics.business_output["bw_util_pct"]     = round(_bw_util, 1)
            metrics.business_output["bw_gbs_used"]     = round(_bw_gbs, 1)
            metrics.business_output["tflops_est"]      = round(_tflops_est, 2)
            metrics.business_output["tflops_util_pct"] = round(_tfu, 1)
            metrics.business_output["p99_ms_est"]      = round(_p99_est, 1)

            # --- Dimension 2: Bottleneck ---
            from .bottleneck_analyzer import classify_bottleneck
            metrics.primary_bottleneck = classify_bottleneck(metrics, total_vram)

            return metrics

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                total_vram = get_total_vram_mb()
                peak = get_gpu_peak_memory_mb()
                metrics.error = f"OOM at batch={batch_size}"
                metrics.primary_bottleneck = "Memory Capacity"
                metrics.peak_memory_mb = peak if peak > 0 else total_vram
                metrics.workload_phase = "prefill+decode"
                metrics.operational_condition = {
                    "num_users": num_users,
                    "context_length": context_length,
                    "batch_size": batch_size,
                    "fits_in_memory": False,
                    "mem_pct": 100.0,
                }
                metrics.business_output = {"load_ms": round(self._load_ms, 2)}
            else:
                metrics.error = str(e)
            return metrics
        except Exception as e:
            metrics.error = str(e)
            return metrics

    def generate_text(self, prompt: str, max_new_tokens: int = 220) -> str:
        """
        Free-form generation using the currently-loaded model — runs entirely on the
        GB10. Used to let the just-benchmarked model narrate its own results on-device.
        Uses the tokenizer's chat template when available so instruct models behave.
        Returns "" if no model is loaded; never raises.
        """
        if self.model is None or self.tokenizer is None:
            return ""
        try:
            tok = self.tokenizer
            if getattr(tok, "chat_template", None):
                enc = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                # Depending on transformers version this is a bare tensor or a BatchEncoding
                if hasattr(enc, "input_ids"):
                    input_ids = enc.input_ids
                elif isinstance(enc, dict):
                    input_ids = enc["input_ids"]
                else:
                    input_ids = enc
            else:
                input_ids = tok(prompt, return_tensors="pt").input_ids
            if torch.cuda.is_available():
                input_ids = input_ids.to(next(self.model.parameters()).device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    pad_token_id=tok.eos_token_id,
                )
            new_tokens = out[0][input_ids.shape[1]:]
            return tok.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            return f"(on-device generation unavailable: {e})"

    def generate_stream(self, prompt: str, system: Optional[str] = None,
                        max_new_tokens: int = 450):
        """
        Token-by-token streaming generation on the GB10 — yields text pieces suitable
        for st.write_stream(). Used by the on-device AI narrator. Yields a single
        notice string on failure rather than raising.
        """
        if self.model is None or self.tokenizer is None:
            yield "⚠ on-device model not loaded"
            return
        try:
            tok = self.tokenizer
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            if getattr(tok, "chat_template", None):
                enc = tok.apply_chat_template(
                    msgs, add_generation_prompt=True, return_tensors="pt")
                if hasattr(enc, "input_ids"):
                    input_ids = enc.input_ids
                elif isinstance(enc, dict):
                    input_ids = enc["input_ids"]
                else:
                    input_ids = enc
            else:
                text = (system + "\n\n" if system else "") + prompt
                input_ids = tok(text, return_tensors="pt").input_ids
            if torch.cuda.is_available():
                input_ids = input_ids.to(next(self.model.parameters()).device)
            attention_mask = torch.ones_like(input_ids)

            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            kwargs = dict(
                input_ids=input_ids, attention_mask=attention_mask, streamer=streamer,
                max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7,
                top_p=0.9, repetition_penalty=1.15, pad_token_id=tok.eos_token_id,
                # The repetition_penalty + top_p processor chain can leave the
                # distribution un-normalized in BF16; without this, multinomial
                # sampling can hit inf/nan and fire a device-side assert that
                # poisons the whole CUDA context. HF explicitly recommends it.
                renormalize_logits=True,
            )

            # Capture any error from the worker thread. A CUDA fault here (e.g. a
            # device-side assert) would otherwise be swallowed, leave the streamer
            # blocked, and only surface on the NEXT CUDA call — making the traceback
            # point at an unrelated place (e.g. unload's synchronize).
            _gen_err: dict = {}

            def _gen():
                try:
                    with torch.no_grad():
                        self.model.generate(**kwargs)
                except Exception as e:          # noqa: BLE001 — surface to the consumer
                    _gen_err["e"] = e
                finally:
                    # Always unblock the consumer loop, even if generate() raised
                    # before the streamer emitted its stop signal.
                    try:
                        streamer.text_queue.put(streamer.stop_signal)
                    except Exception:
                        pass

            thread = threading.Thread(target=_gen, daemon=True)
            thread.start()
            for piece in streamer:
                yield piece
            thread.join()
            if _gen_err:
                e = _gen_err["e"]
                yield f"\n\n⚠ on-device generation error: {type(e).__name__}: {e}"
        except Exception as e:
            yield f"\n\n⚠ on-device generation error: {type(e).__name__}: {e}"

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            # The context may already be in an error state (e.g. a prior device-side
            # assert during generation). Guard both calls so unload — which runs at
            # module scope on benchmark launch — can never crash the whole app.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def benchmark_llm(model_name: str, precisions: list = None, batch_size: int = 1) -> list:
    if precisions is None:
        precisions = ["FP32", "FP16"]
    results = []
    for precision in precisions:
        inference = LLMInference(model_name, precision)
        if inference.load_model():
            metrics = inference.benchmark(batch_size=batch_size)
            results.append(metrics)
        else:
            m = BenchmarkMetrics(model_name, precision)
            m.error = "Failed to load model"
            results.append(m)
        inference.unload()
    return results
