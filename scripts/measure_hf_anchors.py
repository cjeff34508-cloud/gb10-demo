#!/usr/bin/env python3
"""Measure single-stream tok/s for every HF-runnable model, once, and store it.

Why this exists: the TCO engine needs ONE throughput anchor per model, and it
never loads a model to get it. vLLM captures provide anchors for the quantized
checkpoints the HF stack cannot run — but the plain checkpoints the HF stack CAN
run had no per-model anchor at all, so they all fell back to a single number and
their $/MTok columns came out identical. That is not a modelling result, it is a
missing measurement.

This runs each HF-runnable model once, times a real generation, and writes
webui/data/hf_anchors.json. One model resident at a time, same guards as the
Benchmark tab, page cache released after each load.
"""
import json, sys, time
from pathlib import Path

ROOT = Path.home() / "gb10-demo"
sys.path.insert(0, str(ROOT / "webui"))
OUT = ROOT / "webui" / "data" / "hf_anchors.json"
PROMPT = "Explain in two sentences why memory bandwidth limits LLM decode speed."
NEW_TOKENS = 96

from helpers.llm_inference import LLMInference          # noqa: E402
from helpers.tco_engine import BENCHMARKABLE, native_precision  # noqa: E402
from helpers.benchmark_utils import free_cuda_memory    # noqa: E402

def precision_for(hf_id: str) -> str:
    u = hf_id.upper()
    if "NVFP4" in u:
        return "NVFP4"
    return "BF16"

def main():
    only = sys.argv[1:] or list(BENCHMARKABLE)
    data = json.loads(OUT.read_text()) if OUT.exists() else {"anchors": {}}
    for hf_id in only:
        cat = BENCHMARKABLE.get(hf_id)
        if not cat:
            print(f"  skip {hf_id}: not in BENCHMARKABLE")
            continue
        prec = precision_for(hf_id)
        print(f"→ {hf_id}  @{prec}", flush=True)
        inf = LLMInference(hf_id, prec)
        if not inf.load_model():
            print(f"  ✗ load refused: {getattr(inf, 'load_block_reason', '') or '(raised)'}")
            free_cuda_memory(); continue
        try:
            # warm, then time a real generation
            list(inf.generate_stream(PROMPT, max_new_tokens=8))
            t0 = time.perf_counter()
            txt = "".join(inf.generate_stream(PROMPT, max_new_tokens=NEW_TOKENS))
            dt = time.perf_counter() - t0
            ntok = len(inf.tokenizer(txt)["input_ids"]) if inf.tokenizer else NEW_TOKENS
            tps = ntok / dt if dt > 0 else 0.0
            ascii_frac = sum(c.isascii() for c in txt) / max(len(txt), 1)
            data["anchors"][cat] = {
                "hf_id": hf_id, "precision": inf._effective_precision or prec,
                "tok_s": round(tps, 2), "new_tokens": ntok,
                "coherent_ascii_frac": round(ascii_frac, 3),
                "sample": txt.strip()[:120],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "method": "HF eager single stream",
            }
            print(f"  ✓ {tps:.1f} tok/s · ascii {ascii_frac:.2f} · {txt.strip()[:60]!r}")
        except Exception as e:
            print(f"  ✗ generate failed: {type(e).__name__}: {e}")
        finally:
            inf.unload(); free_cuda_memory()
        OUT.write_text(json.dumps(data, indent=1))   # write as we go; resumable
    print(f"\n■ {len(data['anchors'])} anchors in {OUT.name}")
    for k, v in sorted(data["anchors"].items(), key=lambda kv: kv[1]["tok_s"]):
        print(f"  {k:26s} {v['tok_s']:7.1f} tok/s  {v['precision']:14s} ascii {v['coherent_ascii_frac']:.2f}")

if __name__ == "__main__":
    raise SystemExit(main())
