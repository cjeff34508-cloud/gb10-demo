#!/usr/bin/env python
"""Convert a ModelOpt-exported NVFP4 HF checkpoint into a compressed-tensors
`nvfp4-pack-quantized` checkpoint that stock transformers + compressed-tensors
can load natively.

The on-disk packing is already compatible (U8-packed 4-bit weights, FP8_E4M3
per-group scales at group_size 16, FP32 global scale). Only the tensor *names*
and the `quantization_config` schema differ:

    modelopt                 ->  compressed-tensors
    weight (uint8)               weight_packed
    weight_scale (fp8_e4m3)      weight_scale          (unchanged)
    weight_scale_2 (fp32)        weight_global_scale
    input_scale (fp32)           input_global_scale

KV-cache scales (k_scale/v_scale) are dropped for now (FP8 KV cache disabled)
to keep this first correctness pass simple.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import argparse

_ROOT = Path.home() / "gb10-demo"
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--src", required=True,
                 help="ModelOpt checkpoint dir (absolute, or relative to ~/gb10-demo)")
_ap.add_argument("--dst", required=True, help="output dir for the converted checkpoint")
_a = _ap.parse_args()


def _resolve(x: str) -> Path:
    q = Path(x)
    return q if q.is_absolute() else _ROOT / q


SRC = _resolve(_a.src)
DST = _resolve(_a.dst)
if not (SRC / "config.json").exists():
    raise SystemExit(f"no config.json in {SRC}")
if DST.exists() and any(DST.glob("*.safetensors")):
    raise SystemExit(f"{DST} already holds shards — refusing to overwrite")
print(f"src: {SRC}\ndst: {DST}")

RENAME = {
    "weight_scale_2": "weight_global_scale",
    "input_scale": "input_global_scale",
}
DROP_SUFFIXES = {"k_scale", "v_scale"}  # KV-cache quant disabled for this pass

DST.mkdir(parents=True, exist_ok=True)

# First pass: which module bases are quantized (their .weight is uint8)?
quant_bases: set[str] = set()
shards = sorted(SRC.glob("*.safetensors"))
for shard in shards:
    with safe_open(shard, "pt") as h:
        for k in h.keys():
            if k.endswith(".weight") and h.get_slice(k).get_dtype() == "U8":
                quant_bases.add(k[: -len(".weight")])
print(f"quantized linear modules (U8-packed .weight): {len(quant_bases)}")

# A base with weight_scale but NO weight_scale_2 is FP8-quantized, not NVFP4.
# This converter only handles the NVFP4 packing, so flag them loudly rather than
# emitting a checkpoint that is half-translated and claims to be whole.
_have_s2: set[str] = set()
for _sh in shards:
    with safe_open(_sh, "pt") as _h:
        for _k in _h.keys():
            if _k.endswith(".weight_scale_2"):
                _have_s2.add(_k[: -len(".weight_scale_2")])
_missing = quant_bases - _have_s2
if _missing:
    print(f"!! {len(_missing)} U8 modules have no weight_scale_2 (FP8, not NVFP4) — "
          f"e.g. {sorted(_missing)[:3]}")
    print("!! this converter handles NVFP4 only; the result would be mixed. Aborting.")
    raise SystemExit(2)

# Extra ModelOpt-specific tensors stock compressed-tensors has no schema for.
_extra: set[str] = set()
for _sh in shards:
    with safe_open(_sh, "pt") as _h:
        for _k in _h.keys():
            _sx = _k.split(".")[-1]
            if _sx in ("layer_scalar", "per_expert_scale", "scale"):
                _extra.add(_sx)
if _extra:
    print(f"note: passing through non-standard ModelOpt tensors: {sorted(_extra)} "
          f"— if the load fails, these are the first suspects")

# Second pass: rewrite each shard, building the new weight_map.
weight_map: dict[str, str] = {}
for shard in shards:
    out_tensors: dict[str, torch.Tensor] = {}
    with safe_open(shard, "pt") as h:
        for k in h.keys():
            suffix = k.split(".")[-1]
            if suffix in DROP_SUFFIXES:
                continue
            base = k[: -(len(suffix) + 1)]
            nk = k
            t = h.get_tensor(k)
            if base in quant_bases:
                if suffix == "weight":
                    nk = base + ".weight_packed"
                elif suffix in RENAME:
                    nk = base + "." + RENAME[suffix]
                    # modelopt stores a *dequant multiplier* (w = q*scale*scale_2);
                    # compressed-tensors divides by the global scale
                    # (eff = scale/global_scale), so the global scales are reciprocals.
                    t = (1.0 / t.float())
            out_tensors[nk] = t
            weight_map[nk] = shard.name
    save_file(out_tensors, DST / shard.name, metadata={"format": "pt"})
    print(f"wrote {shard.name}: {len(out_tensors)} tensors")

# Index file
total_size = sum((DST / s.name).stat().st_size for s in shards)
(DST / "model.safetensors.index.json").write_text(
    json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
)

# config.json with compressed-tensors NVFP4 schema
cfg = json.loads((SRC / "config.json").read_text())
cfg["quantization_config"] = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "quantization_status": "compressed",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4, "type": "float", "symmetric": True,
                "group_size": 16, "strategy": "tensor_group", "dynamic": False,
            },
            "input_activations": {
                "num_bits": 4, "type": "float", "symmetric": True,
                "group_size": 16, "strategy": "tensor_group", "dynamic": "local",
            },
        }
    },
    "ignore": ["lm_head"],
}
(DST / "config.json").write_text(json.dumps(cfg, indent=2))

# Copy tokenizer / generation config verbatim
for name in ("generation_config.json", "tokenizer_config.json", "tokenizer.json",
             "vocab.json", "added_tokens.json", "special_tokens_map.json",
             "merges.txt", "chat_template.jinja", "chat_template.json",
             # a VLM is unusable without its image processor
             "preprocessor_config.json", "processor_config.json",
             "video_preprocessor_config.json"):
    src = SRC / name
    if src.exists():
        (DST / name).write_bytes(src.read_bytes())

print(f"DONE -> {DST}")
