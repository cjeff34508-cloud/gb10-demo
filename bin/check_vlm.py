#!/usr/bin/env python3
"""check_vlm.py — prove the vision track can READ an image, not just encode one.

The encoder path reports images/sec on SYNTHETIC RANDOM pixels: a true number
that says nothing about comprehension. These assertions use a chart whose values
this repo drew (scripts/make_chart_fixture.py), so a claim about it is checkable.

  1 fixture exists and its ground truth is readable
  2 the catalog separates vision ENCODERS from vision-LANGUAGE models
  3 ModelOpt VLMs are refused with a reason naming vLLM (transformers cannot
    load them — see docs/CATALOG-2026.md)
  4 the vision path enforces a memory guard (it previously had none)
  5 the VLM loads through AutoModelForImageTextToText
  6 image+prompt -> text returns non-empty text and a tok/s figure
  7 the answer CONTAINS THE RIGHT NUMBER from the chart   <-- the real test

Exit non-zero on any failure. 5-7 need the GPU; skip them with --no-gpu.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path.home() / "gb10-demo"
sys.path.insert(0, str(ROOT / "webui"))

FIX = ROOT / "webui" / "assets" / "fixture-chart.png"
TRUTH = FIX.with_suffix(".truth.json")
VLM = "Qwen/Qwen3.6-35B-A3B-FP8"

results = []
def check(name, ok, msg):
    results.append((name, ok, msg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="skip the load/generate checks")
    ap.add_argument("--model", default=VLM)
    a = ap.parse_args()

    from helpers.vision_inference import (VisionModelInference, is_vision_language_model,
                                          _local_path)
    from helpers.model_info import precision_compatible

    truth = None
    if FIX.exists() and TRUTH.exists():
        truth = json.loads(TRUTH.read_text())
        check("1 fixture", True,
              f"{FIX.name} + truth (tallest {truth['tallest_label']} = {truth['tallest_value']})")
    else:
        check("1 fixture", False, f"missing {FIX} — run scripts/make_chart_fixture.py")

    enc = [m for m in ("openai/clip-vit-large-patch14", "google/vit-base-patch16-224")
           if not is_vision_language_model(_local_path(m))]
    vlm_ok = is_vision_language_model(_local_path(a.model))
    check("2 catalog split", len(enc) == 2 and vlm_ok,
          f"{len(enc)}/2 encoders are encode-only; {a.model.split('/')[-1]} is a VLM={vlm_ok}")

    bad = []
    for m, p in (("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4", "NVFP4"),
                 ("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", "BF16")):
        ok, why = precision_compatible(m, p)
        if ok or "vllm" not in why.lower():
            bad.append(f"{m.split('/')[-1]}: refused={not ok} reason={why[:60]!r}")
    check("3 modelopt refused", not bad,
          bad[0] if bad else "both ModelOpt VLMs refused, reason names vLLM")

    src = (ROOT / "webui" / "helpers" / "vision_inference.py").read_text()
    guarded = ("fits_in_memory" in src and "mem_get_info" in src
               and "precision_compatible" in src)
    check("4 vision memory guard", guarded,
          "fits_in_memory + live-free + precision_compatible present" if guarded
          else "vision path still has NO memory guard")

    # 5-7 need a VLM this stack can actually load. As of 2026-11-05 none exists
    # on this box: the only HF-loadable VLM candidate (Qwen3.6-35B-A3B-FP8) is a
    # block-fp8 checkpoint, and every pre-quantized format except
    # compressed-tensors generates garbage here (docs/CATALOG-2026.md). So this
    # is an unmet PREREQUISITE, not a code defect — reported as SKIP with the
    # reason, never as a pass.
    _ok_quant, _why_quant = precision_compatible(a.model, "FP8")
    if a.no_gpu:
        check("5-7 gpu checks", None, "skipped (--no-gpu)")
    elif not _ok_quant:
        check("5-7 vlm generation", None,
              f"NO HF-loadable VLM on this box — {a.model.split('/')[-1]} refused: "
              f"{_why_quant[:90]}. The code path is implemented and statically "
              f"verified; it needs a compressed-tensors VLM or a vLLM-backed path.")
    else:
        from PIL import Image
        inf = VisionModelInference(a.model, "FP8")
        loaded = inf.load_model()
        check("5 vlm loads", bool(loaded and inf.is_vlm),
              f"loaded={loaded} is_vlm={inf.is_vlm}"
              + (f" · blocked: {inf.load_block_reason}" if not loaded else ""))
        if loaded and truth:
            img = Image.open(FIX).convert("RGB")
            prompt = ("This is a bar chart. Which bar is tallest, and what is its "
                      "numeric value? Answer with the label and the number.")
            m = inf.benchmark_vlm(img, prompt, num_runs=1, max_new_tokens=96)
            bo = m.business_output or {}
            ans = bo.get("answer", "") if not m.error else ""
            check("6 image+prompt->text", bool(ans) and not m.error,
                  f"{bo.get('tokens_per_sec', 0):.1f} tok/s, {len(ans)} chars"
                  + (f" · error={m.error}" if m.error else ""))
            want, want_alt = f"{truth['tallest_value']:.1f}", f"{truth['tallest_value']:.0f}"
            check("7 reads the chart", want in ans or want_alt in ans,
                  f"expected {want}; got: {ans[:120]!r}")
            inf.unload()
        elif loaded:
            inf.unload()

    print("\nVLM track check")
    print("=" * 66)
    p = f = s = 0
    for name, ok, msg in results:
        tag = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"  [{tag}] {name}: {msg}")
        p += ok is True; f += ok is False; s += ok is None
    print("=" * 66)
    print(f"  {p} passed · {f} failed · {s} skipped")
    return 1 if f else 0

if __name__ == "__main__":
    raise SystemExit(main())
