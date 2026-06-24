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

SRC = Path("models/llm-models/nvidia--Qwen3-8B-NVFP4")
DST = Path("models/llm-models/nvidia--Qwen3-8B-NVFP4-ct")

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
print(f"quantized linear modules: {len(quant_bases)}")

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
             "vocab.json", "added_tokens.json", "special_tokens_map.json"):
    src = SRC / name
    if src.exists():
        (DST / name).write_bytes(src.read_bytes())

print(f"DONE -> {DST}")
