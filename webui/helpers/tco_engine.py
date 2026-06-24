"""
TCO Engine — Dell HW catalog, workload-based TCO calculation, and HW recommendation.

All prices are approximate list prices (USD) as of 2025-2026.
Power, rack, and networking estimates are typical data-center values.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Per-GPU Spec Reference (single source of truth)
# ---------------------------------------------------------------------------
# Authoritative per-GPU specs. Dell systems below reference these by `gpu_spec`
# and have their aggregate VRAM / memory bandwidth / TFLOPS DERIVED from here
# (per-GPU value × gpus_per_node) by _derive_system_specs(), so the catalog and
# this table can never drift apart.
#
# `mem_bw_gbs` is per-GPU memory bandwidth in GB/s (the decode-relevant ceiling).
# `tflops` is per-GPU Tensor-Core throughput, stored as {"dense", "sparse"} for
# each precision. Where a source gives only the headline (with-sparsity) Hopper
# number, dense is the 2:1 half; where a source gives only a dense Blackwell
# number, sparse is the 2:1 double. FP32/FP64 have no sparsity (dense == sparse).
# TF32 is the tensor-core matmul mode (FP32 storage): it runs at ½ the FP16 rate
# on Hopper/Blackwell, so it is derived as ½ of each part's FP16 dense/sparse.
# Values mirror the supplied Dell/NVIDIA spec sheet exactly where stated.

def _tf(dense: float, sparse: float | None = None) -> dict:
    """Per-precision TFLOPS entry. sparse defaults to 2× dense (or = dense for FP32/64)."""
    return {"dense": dense, "sparse": dense if sparse is None else sparse}

GPU_SPECS: dict[str, dict] = {
    "RTX PRO 6000 BSE": {
        "arch": "Blackwell", "class": "PCIe DW server GPU",
        "mem_gb": 96, "mem_type": "GDDR7", "mem_bw_gbs": 1_800,
        "link": "PCIe Gen5 x16", "nvlink_gbs": 0, "gpu_tdp_w": 300,
        "rt_tflops": 380,  # RT Core ray-tracing throughput (not a tensor precision)
        "tflops": {
            # AI headline "4,000 TOPS" FP4 = with-sparsity; dense is the 2:1 half
            "FP4":  _tf(2_000, 4_000), "FP32": _tf(125), "FP64": _tf(1.97),
        },
        "notes": "Dell positions for inference, fine-tuning, simulation, visual computing. No NVLink in Dell matrix. "
                 "FP32 125 TFLOPS · FP64 1.97 TFLOPS · RT Core 380 TFLOPS · FP4 4,000 TOPS (≈4 PFLOPS, w/ sparsity).",
    },
    "RTX PRO 4500 BSE": {
        "arch": "Blackwell", "class": "PCIe single-slot server GPU",
        "mem_gb": 32, "mem_type": "GDDR7", "mem_bw_gbs": 800,   # 256-bit, 25 Gbps GDDR7
        "link": "PCIe Gen5 x16", "nvlink_gbs": 0, "gpu_tdp_w": 165,
        "rt_tflops": 154,  # RT Core peak
        "tflops": {
            # NVIDIA datasheet headline numbers are "with sparsity"; dense is the 2:1 half.
            # FP4 1.6 PFLOPS · FP8 811 · FP16/BF16 406 · TF32 203 · FP32 51 (no sparsity).
            "FP4":  _tf(800, 1_600), "FP8":  _tf(406, 811),
            "FP16": _tf(203, 406),   "BF16": _tf(203, 406),
            "TF32": _tf(101, 203),   "FP32": _tf(51),
        },
        "notes": "GB203 · 10,496 CUDA cores · 82 RT cores · 32 GB GDDR7 @ 800 GB/s (256-bit) · single-slot "
                 "FHFL passive · 165 W · PCIe Gen5 x16, no NVLink · FP4 1.6 PFLOPS (w/ sparsity).",
    },
    "H100 PCIe": {
        "arch": "Hopper", "class": "PCIe Gen5 board",
        "mem_gb": 80, "mem_type": "HBM2e", "mem_bw_gbs": 2_000,
        "link": "PCIe Gen5", "nvlink_gbs": 0,
        "tflops": {
            "FP8":  _tf(1_600, 3_200), "FP16": _tf(800, 1_600),
            "BF16": _tf(800, 1_600),   "TF32": _tf(400, 800), "FP32": _tf(48),
        },
        "notes": "14,592 FP32 cores · 456 Tensor Cores.",
    },
    "H100 NVL": {
        "arch": "Hopper", "class": "PCIe dual-slot air-cooled",
        "mem_gb": 94, "mem_type": "HBM3", "mem_bw_gbs": 3_900,
        "link": "NVLink 600 GB/s bridge + PCIe Gen5 128 GB/s", "nvlink_gbs": 600,
        "tflops": {
            "FP8":  _tf(1_671, 3_341), "FP16": _tf(835, 1_671),
            "BF16": _tf(835, 1_671),   "TF32": _tf(418, 835), "FP32": _tf(60),
        },
        "notes": "94 GB HBM3 @ 3.9 TB/s — faster memory than the 80 GB SXM5; 2-GPU NVLink islands.",
    },
    "H100 SXM5": {
        "arch": "Hopper", "class": "SXM5 / HGX",
        "mem_gb": 80, "mem_type": "HBM3", "mem_bw_gbs": 3_350,  # Dell matrix rounds to 3.0 TB/s
        "link": "NVLink 900 GB/s + PCIe Gen5 128 GB/s", "nvlink_gbs": 900,
        "tflops": {
            "FP8":  _tf(1_979, 3_958), "FP16": _tf(990, 1_979),
            "BF16": _tf(990, 1_979),   "TF32": _tf(495, 990), "FP32": _tf(67),
        },
        "notes": "16,896 FP32 cores · 528 Tensor Cores. Official BW 3.35 TB/s (Dell matrix rounds to 3.0).",
    },
    "H200 SXM5": {
        "arch": "Hopper", "class": "SXM5 / HGX",
        "mem_gb": 141, "mem_type": "HBM3e", "mem_bw_gbs": 4_800,
        "link": "NVLink 900 GB/s + PCIe Gen5 x16", "nvlink_gbs": 900,
        "tflops": {  # same Hopper compute as H100 SXM5; the uplift is memory capacity/BW
            "FP8":  _tf(1_979, 3_958), "FP16": _tf(990, 1_979),
            "BF16": _tf(990, 1_979),   "TF32": _tf(495, 990), "FP32": _tf(67),
        },
        "notes": "Same Hopper tensor throughput as H100 SXM5; Dell emphasizes the 141 GB / 4.8 TB/s memory uplift.",
    },
    "H200 NVL": {
        "arch": "Hopper", "class": "PCIe dual-slot (NVL)",
        "mem_gb": 141, "mem_type": "HBM3e", "mem_bw_gbs": 4_800,
        "link": "NVLink 900 GB/s bridge + PCIe Gen5 x16", "nvlink_gbs": 900,
        "tflops": {  # Hopper compute, same as H200/H100 SXM5
            "FP8":  _tf(1_979, 3_958), "FP16": _tf(990, 1_979),
            "BF16": _tf(990, 1_979),   "TF32": _tf(495, 990), "FP32": _tf(67),
        },
        "notes": "PCIe NVL variant of H200 — 141 GB HBM3e @ 4.8 TB/s with 900 GB/s NVLink bridge (2-GPU islands).",
    },
    "B200": {
        "arch": "Blackwell", "class": "SXM / HGX",
        "mem_gb": 192, "mem_type": "HBM3E", "mem_bw_gbs": 8_000,
        "link": "NVLink 5, 1.8 TB/s per GPU", "nvlink_gbs": 1_800, "max_power_w": 1_200,
        "tflops": {
            "FP4":  _tf(10_000, 20_000), "FP8":  _tf(5_000, 10_000),
            "FP16": _tf(2_500, 5_000),   "BF16": _tf(2_500, 5_000),  # FP16 derived as FP8/2
            "TF32": _tf(1_250, 2_500),
        },
        "notes": "NVFP4 10 PFLOPS dense / 20 sparse; FP8 5/10 PFLOPS. Max power up to 1,200 W. FP16 derived (FP8÷2).",
    },
    "B300": {
        "arch": "Blackwell Ultra", "class": "SXM / HGX",
        "mem_gb": 288, "mem_type": "HBM3E", "mem_bw_gbs": 8_000,
        "link": "NVLink 5, 1.8 TB/s per GPU", "nvlink_gbs": 1_800, "max_power_w": 1_400,
        "tflops": {
            "FP4":  _tf(15_000, 20_000), "FP8":  _tf(5_000, 10_000),
            "FP16": _tf(2_500, 5_000),   "BF16": _tf(2_500, 5_000),  # FP16 derived as FP8/2
            "TF32": _tf(1_250, 2_500),
        },
        "notes": "NVFP4 15 PFLOPS dense / 20 sparse; FP8 5/10 PFLOPS. Max power up to 1,400 W. FP16 derived (FP8÷2).",
    },
    "GB200": {
        "arch": "Grace Blackwell superchip", "class": "NVL72-class",
        "mem_gb": 192, "mem_type": "HBM3E", "mem_bw_gbs": 8_000,  # per Blackwell GPU
        "link": "NVLink 5 1.8 TB/s per GPU + NVLink-C2C 900 GB/s CPU↔GPU", "nvlink_gbs": 1_800,
        "tflops": {  # per Blackwell GPU (same silicon as B200)
            "FP4":  _tf(10_000, 20_000), "FP8":  _tf(5_000, 10_000),
            "FP16": _tf(2_500, 5_000),   "BF16": _tf(2_500, 5_000),   "TF32": _tf(1_250, 2_500),
        },
        "notes": "Superchip: 372 GB HBM3E / 16 TB/s (2 GPUs). NVL72 system: 13.4 TB HBM3E / 576 TB/s; "
                 "1,440 PFLOPS FP4 (sparse) / 720 PFLOPS FP8 (sparse) across 72 GPUs.",
    },
    "GB300": {
        "arch": "Grace Blackwell Ultra superchip", "class": "NVL72-class",
        "mem_gb": 288, "mem_type": "HBM3E", "mem_bw_gbs": 8_000,  # per Blackwell Ultra GPU
        "link": "NVLink 5 / NVLink switching + ConnectX-8 800 Gb/s", "nvlink_gbs": 1_800,
        "tflops": {  # per Blackwell Ultra GPU (same silicon as B300)
            "FP4":  _tf(15_000, 20_000), "FP8":  _tf(5_000, 10_000),
            "FP16": _tf(2_500, 5_000),   "BF16": _tf(2_500, 5_000),   "TF32": _tf(1_250, 2_500),
        },
        "notes": "Up to 1 TB unified memory per superchip; 30 PFLOPS dense NVFP4 per superchip. "
                 "GB300 NVL72 reaches 1.1 exaFLOPS dense FP4; Dell cites 50× AI-reasoning output vs Hopper.",
    },
    "Vera Rubin / VR200": {
        "arch": "Rubin (planned)", "class": "HGX Rubin NVL8 / Vera Rubin platform",
        "mem_gb": 288, "mem_type": "HBM4", "mem_bw_gbs": 22_000,
        "link": "NVLink 6, 3.6 TB/s per GPU", "nvlink_gbs": 3_600, "planned": True,
        "tflops": {
            # FP4 dense = NVFP4 training peak (35 PFLOPS), sparse = NVFP4 inference peak (50 PFLOPS)
            "FP4":  _tf(35_000, 50_000), "FP8":  _tf(17_500, 35_000),  # FP8/FP6 training 17.5 PFLOPS
            "FP16": _tf(4_000, 8_000),   "BF16": _tf(4_000, 8_000),
            "TF32": _tf(2_000, 4_000),   "FP32": _tf(130),            "FP64": _tf(33),
        },
        "notes": "Planned. NVFP4 inference 50 PFLOPS / training 35 PFLOPS; FP8/FP6 17.5 PFLOPS; "
                 "FP16/BF16 4 PFLOPS; FP32 130 TFLOPS; FP64 33 TFLOPS. HGX Rubin NVL8 ≈ 10× B200 token-factory.",
    },
}

# ---------------------------------------------------------------------------
# Dell System Catalog
# ---------------------------------------------------------------------------

DELL_SYSTEMS: dict[str, dict] = {
    "DGX Spark (GB10 SuperChip)": {
        "category":        "Edge / Workstation",
        "system_price":    8_000,
        "gpus_per_node":   1,
        "gpu_spec":        None,          # GB10 not in supplied spec sheet — keep explicit values
        "gpu_model":       "GB10 (Grace-Blackwell)",
        "vram_gb":         128,          # unified LPDDR5X
        "gpu_bw_gbs":      273,          # real LPDDR5X memory BW (~273 GB/s), not NVLink-C2C 900
        "tflops_fp16":     500,
        "tflops_fp8":      1_000,
        "tflops_fp4":      1_000,        # Blackwell FP4 hardware-accelerated
        "system_tdp_w":    60,
        "rack_u":          0,            # desktop / tabletop
        "nw_gbps":         10,
        "nvlink":          False,
        "sxm":             False,
        "gpu_link":        "none",       # single GPU — no in-box GPU-to-GPU fabric
        "link_label":      "Single GPU",
        "net":             "10 GbE",
        "color":           "#76B900",    # NVIDIA green
        "notes":           "Current demo HW · 128 GB unified · lowest $/watt",
        "best_for":        ["LLM ≤32B FP16", "FinTech edge", "VLM batch ≤64"],
    },
    "Dell PowerEdge XE7745 (2-8× RTX PRO 4500 32GB)": {
        "category":        "Blackwell Gen",
        "gpus_per_node":   8,            # max config; PCIe box — can be partially populated
        "flexible_gpus":   True,         # populate 2..8 GPUs; cost/power scale with count
        "min_gpus":        2,
        "max_gpus":        8,
        "gpu_spec":        "RTX PRO 4500 BSE",  # 32 GB GDDR7 @ 800 GB/s — derives per-GPU VRAM / BW / TFLOPS
        "gpu_model":       "RTX PRO 4500 BSE 32GB (GDDR7)",
        # Cost / power decompose into a fixed chassis + per-GPU so partial builds price fairly.
        "chassis_price":   120_000,      # dual Xeon, RAM, 2× CX-7, PSUs, NVMe (0-GPU base)
        "gpu_price":       7_200,        # per RTX PRO 4500 Blackwell Server Edition (−20%)
        "system_price":    177_600,      # = chassis + 8 × gpu (full config, catalog reference)
        "chassis_tdp_w":   1_100,        # platform base (0-GPU)
        "gpu_tdp_w":       165,          # per RTX PRO 4500
        "system_tdp_w":    2_420,        # = chassis + 8 × 165 W (full config)
        "rack_u":          4,
        "nw_gbps":         800,          # 2× ConnectX-7 (400G each) for node-to-node
        "nvlink":          False,        # No NVLink on RTX PRO 4500
        "sxm":             False,
        "gpu_link":        "pcie",       # RTX PRO 4500: PCIe Gen5 only
        "link_label":      "PCIe Gen5",
        "net":             "2× ConnectX-7 400G IB",
        "color":           "#00A4A4",
        "notes":           "Flexible 4U PCIe box · 2–8× RTX PRO 4500 (single-slot, 165 W) · 32 GB GDDR7 @ "
                           "800 GB/s/GPU · no NVLink, PCIe Gen5 spanning · partially populated to fit the "
                           "workload (cost/power scale per GPU) · FP4 1.6 PFLOPS (sparse) per GPU · Blackwell gen",
        "best_for":        ["Right-sized inference", "VLM/CNN serving", "Small-/mid-LLM multi-user"],
    },
    "Dell PowerEdge XE7740 (2-8× RTX PRO 6000 BSE 96GB)": {
        "category":        "Blackwell Gen",
        "gpus_per_node":   8,            # max config; PCIe box — can be partially populated
        "flexible_gpus":   True,         # populate 2..8 GPUs; cost/power scale with count
        "min_gpus":        2,
        "max_gpus":        8,
        "gpu_spec":        "RTX PRO 6000 BSE",  # 96 GB GDDR7 @ 1.8 TB/s — derives per-GPU VRAM / BW / TFLOPS
        "gpu_model":       "RTX PRO 6000 BSE 96GB (GDDR7)",
        "chassis_price":   120_000,      # dual Xeon, RAM, 2× CX-7, PSUs, NVMe (0-GPU base)
        "gpu_price":       35_000,       # per RTX PRO 6000 Blackwell Server Edition
        "system_price":    400_000,      # = chassis + 8 × gpu (full config, catalog reference)
        "chassis_tdp_w":   1_100,        # platform base (0-GPU)
        "gpu_tdp_w":       300,          # per RTX PRO 6000 BSE
        "system_tdp_w":    3_500,        # = chassis + 8 × 300 W (full config)
        "rack_u":          4,
        "nw_gbps":         800,          # 2× ConnectX-7 for node-to-node
        "nvlink":          False,        # No NVLink in RTX PRO 6000 BSE
        "sxm":             False,
        "gpu_link":        "pcie",       # RTX PRO 6000 BSE: PCIe Gen5 only
        "link_label":      "PCIe Gen5",
        "net":             "2× ConnectX-7 400G IB",
        "color":           "#76B900",    # NVIDIA green
        "notes":           "Flexible 4U PCIe box · 2–8× RTX PRO 6000 BSE · 96 GB GDDR7 @ 1.8 TB/s/GPU · no "
                           "NVLink — compute-dense workloads over PCIe Gen5 · partially populated to fit the "
                           "workload (cost/power scale per GPU) · FP4 4,000 TOPS (sparse) per GPU · Blackwell gen",
        "best_for":        ["NVFP4 inference", "LLM ≤70B FP16", "Compute-heavy workloads"],
    },
    "Dell PowerEdge XE9640 (4× H100 SXM5 80GB)": {
        "category":        "Current Gen",
        "system_price":    260_000,
        "gpus_per_node":   4,
        "gpu_spec":        "H100 SXM5",  # derives VRAM / BW / TFLOPS
        "gpu_model":       "H100 SXM5 80GB (HBM3)",
        "vram_gb":         320,          # 4 × 80 GB HBM3
        "gpu_bw_gbs":      13_400,       # 4 × 3,350 GB/s (3.35 TB/s/GPU)
        "tflops_fp16":     3_960,        # 4 × 990 dense BF16/FP16 Tensor Core
        "tflops_fp8":      7_916,        # 4 × 1,979 dense FP8 Tensor Core
        "tflops_fp4":      0,            # Hopper — no FP4
        "system_tdp_w":    5_500,        # 4 × ~700 W SXM5 + platform
        "rack_u":          2,
        "nw_gbps":         400,          # 1× ConnectX-7 — limited node-to-node scale-out
        "nvlink":          True,
        "sxm":             True,
        "gpu_link":        "nvlink4",    # SXM5 NVLink-4 — fast tensor-parallel
        "link_label":      "NVLink-4 (SXM5)",
        "net":             "1× ConnectX-7 400G IB",
        "color":           "#0090C0",
        "notes":           "4× H100 80GB SXM5 · HBM3 3.35 TB/s/GPU · NVLink-4 — fast memory & links in compact 2U form factor; single CX-7 → weak multi-node scale-out",
        "best_for":        ["LLM ≤32B FP16", "Low-latency serving", "FinTech HPC"],
    },
    "Dell PowerEdge XE9640 (4× H200 SXM 141GB)": {
        "category":        "Current Gen",
        "system_price":    390_000,
        "gpus_per_node":   4,
        "gpu_spec":        "H200 SXM5",  # derives VRAM / BW / TFLOPS
        "gpu_model":       "H200 SXM 141GB",
        "vram_gb":         564,          # 4 × 141 GB HBM3e
        "gpu_bw_gbs":      19_200,       # 4 × 4,800 GB/s (4.8 TB/s/GPU)
        "tflops_fp16":     3_960,        # 4 × 990 dense BF16/FP16 Tensor Core
        "tflops_fp8":      7_916,        # 4 × 1,979 dense FP8 Tensor Core
        "tflops_fp4":      0,            # Hopper — no FP4
        "system_tdp_w":    6_500,        # 4 × ~850 W SXM5 + platform
        "rack_u":          2,
        "nw_gbps":         400,          # 1× ConnectX-7 — limited node-to-node scale-out
        "nvlink":          True,
        "sxm":             True,
        "gpu_link":        "nvlink4",    # SXM5 NVLink-4 — fast tensor-parallel
        "link_label":      "NVLink-4 (SXM5)",
        "net":             "1× ConnectX-7 400G IB",
        "color":           "#005090",
        "notes":           "4× H200 141GB SXM · HBM3e 4.8 TB/s/GPU · 564 GB VRAM total · NVLink-4 — best per-GPU memory in 2U form factor; 70B FP16 fits in one node; single CX-7 → weak multi-node scale-out",
        "best_for":        ["LLM 70B FP16", "Large context (32K+)", "FinTech HPC"],
    },

    "Dell PowerEdge XE9780 (8× B200 SXM)": {
        "category":        "Blackwell Gen",
        "system_price":    780_000,
        "gpus_per_node":   8,
        "gpu_spec":        "B200",       # derives VRAM / BW / TFLOPS (NVFP4 10/20 PFLOPS, FP8 5/10)
        "gpu_model":       "B200 SXM 192GB",
        "vram_gb":         1_536,        # 8 × 192 GB HBM3e
        "gpu_bw_gbs":      64_000,       # 8 × 8,000 GB/s
        "tflops_fp16":     14_400,       # 8 × 1,800
        "tflops_fp8":      28_800,
        "tflops_fp4":      57_600,       # 8 × 7,200 — HW-accelerated
        "system_tdp_w":    14_400,
        "rack_u":          8,
        "nw_gbps":         6_400,        # 8× ConnectX-8 (800G each) — top-tier multi-node scale-out
        "nvlink":          True,
        "sxm":             True,
        "gpu_link":        "nvlink5",    # SXM NVLink-5 (1.8 TB/s)
        "link_label":      "NVLink-5 (SXM)",
        "net":             "8× ConnectX-8 800G IB",
        "color":           "#E87722",
        "notes":           "Blackwell + NVLink 5.0 · FP4 HW accel · ~29× memory BW vs GB10",
        "best_for":        ["NVFP4 inference", "LLM 70B+ any precision", "FinTech Black-Scholes"],
    },
    "Dell PowerEdge XE9780 (8× B300 SXM)": {
        "category":        "Blackwell Gen",
        "system_price":    950_000,      # est. list — Blackwell Ultra premium over the B200 build
        "gpus_per_node":   8,
        "gpu_spec":        "B300",       # same XE9780 chassis, B300 (Blackwell Ultra) GPUs
        "gpu_model":       "B300 SXM 288GB",
        "vram_gb":         2_304,        # 8 × 288 GB HBM3E (derived)
        "gpu_bw_gbs":      64_000,       # 8 × 8,000 GB/s (derived)
        "system_tdp_w":    16_000,       # 8 × ~1,400 W GPU + platform
        "rack_u":          8,
        "nw_gbps":         6_400,        # 8× ConnectX-8 (800G each) — top-tier multi-node scale-out
        "nvlink":          True,
        "sxm":             True,
        "gpu_link":        "nvlink5",    # SXM NVLink-5 (1.8 TB/s)
        "link_label":      "NVLink-5 (SXM)",
        "net":             "8× ConnectX-8 800G IB",
        "color":           "#C8102E",    # deeper red — top Blackwell Ultra tier
        "notes":           "Same XE9780 SXM chassis as the B200 build, with B300 (Blackwell Ultra) GPUs · "
                           "2.3 TB VRAM · NVFP4 15 PFLOPS dense/GPU · 288 GB HBM3E for the largest models / context",
        "best_for":        ["NVFP4 inference at scale", "LLM 405B FP8 (single node)", "Long-context serving"],
    },
    "Dell NVL72 (GB200 NVL72 Rack)": {
        "category":        "HPC / AI Factory",
        "system_price":    5_200_000,
        "gpus_per_node":   72,           # atomic unit: one full 72-GPU NVLink rack
        "gpu_spec":        "GB200",      # derives VRAM / BW / TFLOPS (per-GPU B200 silicon × 72)
        "gpu_model":       "GB200 192GB",
        "vram_gb":         13_824,       # 72 × 192 GB
        "gpu_bw_gbs":      576_000,      # 72 × 8,000 GB/s
        "system_tdp_w":    120_000,
        "rack_u":          42,           # full rack
        "nw_gbps":         25_600,
        "nvlink":          True,
        "sxm":             True,
        "gpu_link":        "nvlink-switch",  # all 72 GPUs in one NVLink switch domain
        "link_label":      "NVLink Switch (72-GPU rack)",
        "net":             "NVLink rack fabric",
        "rack_unit":       True,         # billed only in whole 72-GPU racks
        "color":           "#00A4E4",
        "notes":           "Rack-scale unit — minimum 72 GPUs · 13.8 TB VRAM · single NVLink domain (no slow cross-node hops within the rack)",
        "best_for":        ["405B+ models", "Hyperscale LLM training", "Enterprise AI platform"],
    },
    "Dell NVL72 (GB300 NVL72 Rack)": {
        "category":        "HPC / AI Factory",
        "system_price":    6_000_000,
        "gpus_per_node":   72,           # atomic unit: one full 72-GPU NVLink rack
        "gpu_spec":        "GB300",      # derives VRAM / BW / TFLOPS (per-GPU B300 silicon × 72)
        "gpu_model":       "GB300 288GB",
        "vram_gb":         20_736,       # 72 × 288 GB HBM3E (derived)
        "gpu_bw_gbs":      576_000,      # 72 × 8,000 GB/s (derived)
        "system_tdp_w":    140_000,      # Blackwell Ultra — higher per-GPU power than GB200
        "rack_u":          42,           # full rack
        "nw_gbps":         28_800,       # ConnectX-8 800G fabric
        "nvlink":          True,
        "sxm":             True,
        "gpu_link":        "nvlink-switch",  # all 72 GPUs in one NVLink switch domain
        "link_label":      "NVLink Switch (72-GPU rack)",
        "net":             "NVLink rack fabric + ConnectX-8 800G",
        "rack_unit":       True,         # billed only in whole 72-GPU racks
        "color":           "#0072CE",
        "notes":           "Blackwell Ultra rack — minimum 72 GPUs · 20.7 TB VRAM · NVFP4 15 PFLOPS dense/GPU · "
                           "single NVLink domain · 1.1 exaFLOPS dense FP4 · Dell cites 50× AI-reasoning output vs Hopper",
        "best_for":        ["Trillion-param inference", "AI-reasoning factories", "Hyperscale training"],
    },
}

# ---------------------------------------------------------------------------
# Derive per-node specs from the GPU_SPECS reference
# ---------------------------------------------------------------------------
# Each system that names a `gpu_spec` has its aggregate VRAM, memory bandwidth,
# and TFLOPS (node-wide dense + sparse) computed as per-GPU value × gpus_per_node.
# This keeps DELL_SYSTEMS and GPU_SPECS in lockstep — and is what corrects the
# H100 NVL / 94GB memory (HBM3 @ 3.9 TB/s, not HBM2 @ 2.0). Systems without a
# `gpu_spec` (e.g. GB10) keep their explicit values. The flat `tflops_fp16/8/4`
# keys are preserved (set to the node-wide DENSE figure) for backward compat.

def aggregate_tflops(gpu_spec_name: str, gpus: int) -> dict[str, dict]:
    """Node-wide {precision: {dense, sparse}} = per-GPU TFLOPS × gpus."""
    spec = GPU_SPECS.get(gpu_spec_name, {})
    out: dict[str, dict] = {}
    for prec, v in spec.get("tflops", {}).items():
        out[prec] = {"dense": v["dense"] * gpus, "sparse": v["sparse"] * gpus}
    return out


def _derive_system_specs() -> None:
    for sys in DELL_SYSTEMS.values():
        key = sys.get("gpu_spec")
        if not key or key not in GPU_SPECS:
            # No reference GPU — synthesize a nested tflops view from the flat keys
            # (dense only; sparsity unknown) so every system has a uniform shape.
            sys.setdefault("tflops", {
                "FP16": {"dense": sys.get("tflops_fp16", 0), "sparse": None},
                "FP8":  {"dense": sys.get("tflops_fp8", 0),  "sparse": None},
                "FP4":  {"dense": sys.get("tflops_fp4", 0),  "sparse": None},
            })
            continue
        spec = GPU_SPECS[key]
        n    = max(sys.get("gpus_per_node", 1), 1)
        sys["vram_gb"]    = int(round(spec["mem_gb"] * n))
        sys["gpu_bw_gbs"] = int(round(spec["mem_bw_gbs"] * n))
        agg = aggregate_tflops(key, n)
        sys["tflops"]      = agg
        sys["tflops_fp16"] = agg.get("FP16", {}).get("dense", 0)
        sys["tflops_fp8"]  = agg.get("FP8", {}).get("dense", 0)
        sys["tflops_fp4"]  = agg.get("FP4", {}).get("dense", 0)


_derive_system_specs()

# ---------------------------------------------------------------------------
# Interconnect efficiency model
# ---------------------------------------------------------------------------
# Tensor-/pipeline-parallel decode loses throughput to GPU-to-GPU communication.
# How much depends on the fabric: NVLink (SXM) is fast, PCIe is slow, and crossing
# a node boundary over InfiniBand (ConnectX-7) is slower still. Each entry is
# (penalty per extra GPU in the parallel group, efficiency floor).
_LINK_PENALTY: dict[str, tuple[float, float]] = {
    "nvlink5":       (0.05, 0.78),   # SXM NVLink-5 (B200) — best
    "nvlink4":       (0.06, 0.72),   # SXM5 NVLink-4 (H100/H200)
    "nvlink-switch": (0.04, 0.80),   # NVL72 rack switch fabric
    "nvlink-bridge": (0.11, 0.55),   # NVL PCIe cards, 2-GPU NVLink islands
    "pcie":          (0.16, 0.42),   # pure PCIe Gen5 spanning — worst
    "none":          (0.00, 1.00),   # single GPU, no spanning
}
# Inter-node penalty per extra node a copy spans. It scales with how much
# node-to-node NIC bandwidth the platform has: one ConnectX-7 (400G) is the
# baseline unit; more cards (e.g. 8× on the XE9680, 1× on the XE9640) shrink the
# per-hop loss. NVLink-switch racks (NVL72) keep a copy in one domain → no hit.
_CROSSNODE_BASE_PENALTY = 0.30   # per-hop loss with a single 400G ConnectX-7
_CROSSNODE_FLOOR        = 0.30
_REF_NIC_GBPS           = 400    # one ConnectX-7 = 1 "NIC unit"


def interconnect_efficiency(sys: dict, gpus_per_model: int, nodes_per_copy: int):
    """Return (intra_eff, inter_eff, parallel_eff) for one model copy.

    intra_eff  — loss from tensor-parallel over the in-box GPU fabric. SXM/NVLink is
                 fast; PCIe boxes (XE7745/XE7740) route GPU-to-GPU over PCIe Gen5 / CX
                 and pay the steep 'pcie' penalty.
    inter_eff  — additional loss when a copy spans node boundaries, eased by more
                 ConnectX cards (nw_gbps). NVL72 racks stay in one NVLink domain → no hit.
    parallel_eff = intra_eff × inter_eff (1.0 for a model that lives on one GPU).
    """
    gpus_in_node = min(gpus_per_model, max(sys["gpus_per_node"], 1))
    alpha, floor = _LINK_PENALTY.get(sys.get("gpu_link", "pcie"), (0.16, 0.42))
    intra_eff = max(floor, 1.0 - alpha * max(0, gpus_in_node - 1))

    if nodes_per_copy > 1 and not sys.get("rack_unit"):
        nic_units = max(1.0, sys.get("nw_gbps", _REF_NIC_GBPS) / _REF_NIC_GBPS)
        per_hop = _CROSSNODE_BASE_PENALTY / nic_units   # more CX cards → less loss/hop
        inter_eff = max(_CROSSNODE_FLOOR, 1.0 - per_hop * (nodes_per_copy - 1))
    else:
        inter_eff = 1.0
    return intra_eff, inter_eff, round(intra_eff * inter_eff, 4)

# ---------------------------------------------------------------------------
# Extended Model Catalog (includes models too large for GB10)
# ---------------------------------------------------------------------------

MODEL_CATALOG: dict[str, dict] = {
    # Small
    "TinyLlama-1.1B":           {"params_b":    1.1, "category": "LLM",  "type": "decoder"},
    "Llama-3.2-1B":             {"params_b":    1.0, "category": "LLM",  "type": "decoder"},
    "Llama-3.2-3B":             {"params_b":    3.2, "category": "LLM",  "type": "decoder"},
    # Medium
    "Mistral-7B":               {"params_b":    7.0, "category": "LLM",  "type": "decoder"},
    "Qwen2.5-7B":               {"params_b":    7.0, "category": "LLM",  "type": "decoder"},
    "Llama-3.1-8B":             {"params_b":    8.0, "category": "LLM",  "type": "decoder"},
    "nvidia/Qwen3-8B-NVFP4":    {"params_b":    8.0, "category": "LLM",  "type": "decoder"},
    # Large
    "Phi-4 (14B)":              {"params_b":   14.0, "category": "LLM",  "type": "decoder"},
    "Qwen2.5-14B":              {"params_b":   14.0, "category": "LLM",  "type": "decoder"},
    "Mixtral-8x7B":             {"params_b":   46.7, "category": "LLM",  "type": "moe"},
    "Qwen2.5-32B":              {"params_b":   32.0, "category": "LLM",  "type": "decoder"},
    # Extra-large (need multi-GPU or large HW)
    "Llama-3.3-70B":            {"params_b":   70.0, "category": "LLM",  "type": "decoder"},
    "Qwen2.5-72B":              {"params_b":   72.0, "category": "LLM",  "type": "decoder"},
    # Hyperscale (multi-node only)
    "Llama-3.1-405B":           {"params_b":  405.0, "category": "LLM",  "type": "decoder"},
    "Llama-3.1-405B (FP8)":     {"params_b":  405.0, "category": "LLM",  "type": "decoder"},
    # VLM / CNN
    "CLIP ViT-L/14":            {"params_b":    0.4, "category": "VLM",  "type": "encoder"},
    "ResNet-50":                {"params_b":    0.03,"category": "CNN",  "type": "encoder"},
    "EfficientNet-B4":          {"params_b":    0.02,"category": "CNN",  "type": "encoder"},
}

_BYTES_PER_PARAM = {
    "FP64": 8.0, "FP32": 4.0, "TF32": 4.0, "FP16": 2.0, "BF16": 2.0,  # TF32 stores as FP32 (4B)
    "INT8": 1.0, "FP8": 1.0, "FP4": 0.5, "NVFP4": 0.5,
}

GB10_BW_GBS = 273  # GB10 real LPDDR5X unified-memory bandwidth (NOT the 900 GB/s
                   # NVLink-C2C link, and not "4 TB/s"). This is the decode-relevant
                   # ceiling and the baseline all systems scale against.

# ---------------------------------------------------------------------------
# FinTech / bandwidth-bound: Monte-Carlo risk & pricing "test data"
# ---------------------------------------------------------------------------
# Memory-bound MC (VaR, Heston, basket pricing) is the canonical bandwidth-bound
# quant workload: a resident batch of path-state is swept every timestep, so
# throughput (paths/sec) is set by memory bandwidth, and the working set must fit
# in VRAM. Unlike LLM decode there is no KV cache, and paths are embarrassingly
# parallel — they shard across GPUs/nodes with ~linear scaling and no NVLink need.
# Per-path state-byte presets are the "test data" complexity selector.
MC_PATH_PRESETS: dict[str, dict] = {
    "GBM — single asset":          {"bytes":   64, "desc": "1 underlying, lognormal"},
    "Heston — stochastic vol":     {"bytes":  256, "desc": "price + variance + RNG state"},
    "Basket — 10 correlated":      {"bytes": 1024, "desc": "10-asset correlated paths"},
    "Portfolio VaR — 250 factors": {"bytes": 4096, "desc": "full risk-factor vector"},
}
# Fraction of peak memory bandwidth a tuned memory-bound MC kernel actually sustains.
MC_BW_EFF = 0.80

# Memory-bandwidth contention when several whole model copies share ONE physical
# GPU. k co-resident copies each re-read their own weights from the same HBM/LPDDR,
# so they cannot each enjoy the full per-GPU bandwidth — without this, a big-VRAM /
# low-bandwidth part (e.g. GB10's 128 GB @ 273 GB/s) gets "free" aggregate just by
# packing more copies in, which physics doesn't allow. We model aggregate per-GPU
# throughput as scaling ∝ k**_COPY_BW_SHARE_EXP, so PER-COPY speed scales
# ∝ k**(_COPY_BW_SHARE_EXP - 1):
#   EXP = 1.0 → no contention (old linear behaviour, aggregate ∝ k)
#   EXP = 0.0 → pure 1/k (aggregate capped at the bandwidth ceiling — the floor for
#               fully independent copies that share nothing)
#   EXP = 0.5 → sqrt: aggregate ∝ √k. Credits the partial weight-amortisation a real
#               batched server gets, while killing the linear free-lunch. Default.
_COPY_BW_SHARE_EXP = 0.5

# Fleet-coordination overhead: a deployment of N nodes is NOT N× a single node. It
# needs more networking (switches, cabling, spine), orchestration, monitoring,
# spares/redundancy, and floor/power distribution — and a stack of independent
# boxes gets no cross-node batching. So a 200-desktop GB10 fleet shouldn't cost-model
# like one unit ×200. We add a CapEx overhead that grows with node count, on TOP of
# the flat add_infra_pct. Logarithmic: mild at datacenter scale (a handful of nodes),
# real for sprawl (hundreds), capped so it never runs away. Applies to every system
# by its node count (fair — it's about how many boxes you must wire up and operate).
#   overhead = min(_FLEET_COORD_CAP, _FLEET_COORD_PER_DOUBLING × log2(nodes))
# At 0.08/doubling: 1 node +0% · 6 +21% · 12 +29% · 25 +37% · 100 +53% · 200 +61%.
_FLEET_COORD_PER_DOUBLING = 0.08
_FLEET_COORD_CAP          = 0.80

# ---------------------------------------------------------------------------
# TCO Data Classes
# ---------------------------------------------------------------------------

@dataclass
class NodeConfig:
    system_name:    str
    num_nodes:      int
    gpus_total:     int
    vram_total_gb:  float
    model_fits:     bool
    gpus_per_model: int       # GPUs needed to hold one model copy
    model_copies:   int       # simultaneous model copies across cluster
    max_sessions:   int       # concurrent user sessions


@dataclass
class TCOResult:
    system_name:        str
    num_nodes:          int
    capex_usd:          float
    annual_power_usd:   float
    tco_usd:            float          # CapEx + power over amort_years
    cost_per_user:      float          # tco / num_users
    cost_per_mtok:      Optional[float]
    predicted_tps:      float
    gpu_util_pct:       float
    fits_single_node:   bool
    recommendation:     str            # "Optimal" | "Viable" | "Oversized" | "Cannot fit"
    rec_reason:         str
    color:              str
    # Performance / capacity descriptors (set by calculate_tco)
    mem_bw_tbs:         float = 0.0     # per-GPU memory bandwidth, TB/s
    vram_total_gb:      float = 0.0     # total VRAM per node, GB
    tps_per_user:       float = 0.0     # delivered tok/s per concurrent user
    bw_contention:      float = 1.0     # per-user BW factor from copies sharing a GPU (≤1)
    fleet_overhead_pct: float = 0.0     # extra CapEx fraction from node-count coordination
    # Interconnect / model-parallel scaling
    gpus_per_node:      int = 0
    gpus_per_copy:      int = 1         # GPUs one model copy spans
    nodes_per_copy:     int = 1         # nodes one model copy spans (>1 = multi-node)
    gpus_total:         int = 0         # total GPUs across the whole solution
    link_label:         str = ""        # GPU fabric (NVLink-4 / PCIe / …)
    parallel_eff:       float = 1.0     # combined intra+inter parallel efficiency
    unit_price:         float = 0.0     # per-node (or per-rack) list price
    # FinTech Monte-Carlo sizing (set by calculate_tco_montecarlo)
    paths_per_sec:      float = 0.0     # achievable MC paths/sec from provisioned GPUs
    cost_per_bpaths:    Optional[float] = None  # $ / billion paths over the amort window
    working_set_gb:     float = 0.0     # resident path-state that must fit in VRAM
    # Combined TCO+Perf rating (set by assign_ratings across a peer group)
    perf_score:         float = 0.0     # 0..1 blend of throughput + cost efficiency
    rating:             str = ""        # "Best" | "Better" | "Good" | "Viable" | "Not Viable"
    rating_reason:      str = ""


# ---------------------------------------------------------------------------
# Core Calculation
# ---------------------------------------------------------------------------

# KV-cache footprint, as a fraction of model weights, per token of sequence.
# Calibrated so a long-context run (~300K tokens) adds roughly the model's own
# weight again in KV cache (~1×), matching real GQA-era 70B-class behaviour. To
# first order this fraction is model-size-independent: both KV and weights scale
# ~linearly with parameters. This lets the TCO model meaningfully stress 300K+
# context (the cap only bites in the multi-million-token regime).
_KV_FRAC_PER_TOKEN = 3.3e-6      # ≈1.0× weights at 300K tokens
_KV_FRAC_CAP       = 8.0         # KV up to 8× weights (multi-million-token ceiling)


def model_memory_gb(model_name: str, precision: str, batch_size: int = 1,
                    context_len: int = 512) -> float:
    """Estimated GPU memory for a model at given precision + runtime conditions."""
    info  = MODEL_CATALOG.get(model_name, {})
    p_b   = info.get("params_b", 7.0)
    bpp   = _BYTES_PER_PARAM.get(precision.upper(), 4.0)
    raw   = p_b * bpp
    # Fixed activation / runtime overhead (weights-relative).
    base_oh      = 0.08
    batch_factor = min(0.003 * batch_size, 0.20)
    # KV cache grows ~linearly with sequence length and batch — the dominant
    # term for long-context serving (e.g. 256K–512K tokens).
    kv_frac      = min(_KV_FRAC_PER_TOKEN * context_len * batch_size, _KV_FRAC_CAP)
    overhead     = base_oh + batch_factor + kv_frac
    return raw * (1.0 + overhead)


def gpus_needed_for_model(model_gb: float, vram_per_gpu: float) -> int:
    """Minimum GPUs (power-of-2) to hold one model copy."""
    if vram_per_gpu >= model_gb:
        return 1
    return min(int(2 ** math.ceil(math.log2(model_gb / vram_per_gpu))), 128)


def node_price(sys: dict, gpus: int) -> float:
    """List price of one node populated with `gpus` GPUs.

    Flexible (PCIe) boxes price as a fixed chassis + per-GPU, so a partial build
    (e.g. 3 or 5 GPUs) costs less than a full one. Fixed-population systems
    (SXM / rack) always return their full `system_price`.
    """
    if sys.get("flexible_gpus"):
        return sys["chassis_price"] + gpus * sys["gpu_price"]
    return sys["system_price"]


def node_tdp_w(sys: dict, gpus: int) -> float:
    """Power draw (W) of one node populated with `gpus` GPUs (chassis + per-GPU
    for flexible boxes; full `system_tdp_w` for fixed-population systems)."""
    if sys.get("flexible_gpus"):
        return sys["chassis_tdp_w"] + gpus * sys.get("gpu_tdp_w", 0)
    return sys["system_tdp_w"]


def effective_node_gpus(sys: dict, gpus_per_copy: int, copies_per_gpu: int,
                        num_users: int) -> int:
    """How many GPUs to populate in one node for this workload.

    SXM / rack systems are always fully populated (`gpus_per_node`). PCIe boxes
    marked ``flexible_gpus`` are populated with the FEWEST GPUs in
    [min_gpus, max_gpus] that still serve the workload — so a job that fits on 3
    or 5 GPUs uses (and is billed for) exactly that many, capped at max_gpus.
    """
    full = max(sys.get("gpus_per_node", 1), 1)
    if not sys.get("flexible_gpus"):
        return full
    min_g = sys.get("min_gpus", 2)
    max_g = sys.get("max_gpus", full)
    if gpus_per_copy >= max_g:
        return max_g                          # one copy already fills (or spans) a full node
    if gpus_per_copy > 1:
        ideal = gpus_per_copy * max(num_users, 1)     # whole copies, each spanning several GPUs
    else:
        ideal = math.ceil(max(num_users, 1) / max(copies_per_gpu, 1))
    ideal = max(ideal, gpus_per_copy, min_g)
    return min(ideal, max_g)


def scale_throughput(
    gb10_tps: float,
    target_bw_gbs: float,
    gpus_in_target: int,
    gb10_bw_gbs: float = GB10_BW_GBS,
) -> float:
    """
    Scale measured GB10 throughput to a target system.
    LLM decode is memory-bandwidth bound → throughput scales with BW ratio.
    Multi-GPU NVLink efficiency: -8% per GPU beyond the first (floor 70%).
    """
    bw_ratio    = target_bw_gbs / max(gb10_bw_gbs, 1)
    nvlink_eff  = max(0.70, 1.0 - 0.08 * max(0, gpus_in_target - 1))
    return gb10_tps * bw_ratio * nvlink_eff


def calculate_tco(
    system_name:    str,
    model_name:     str,
    precision:      str,
    num_users:      int,
    gb10_tps:       float,
    output_toks:    int  = 256,
    context_len:    int  = 512,
    amort_years:    int  = 3,
    power_rate:     float = 0.12,
    add_infra_pct:  float = 0.15,     # networking + storage overhead
) -> TCOResult:
    sys  = DELL_SYSTEMS.get(system_name)
    if not sys:
        raise ValueError(f"Unknown system: {system_name}")

    full_gpus      = max(sys["gpus_per_node"], 1)
    vram_per_gpu   = sys["vram_gb"] / full_gpus
    mem_bw_tbs     = (sys["gpu_bw_gbs"] / full_gpus) / 1000.0
    # Per-session footprint grows with the tokens it must hold in the KV cache:
    # context + generated output. Longer outputs (e.g. 4096) consume more VRAM,
    # so capacity becomes a real performance/cost lever, not just bandwidth.
    eff_ctx        = context_len + output_toks
    m_gb           = model_memory_gb(model_name, precision, batch_size=1, context_len=eff_ctx)
    gpus_per_model = gpus_needed_for_model(m_gb, vram_per_gpu)
    # Viability is judged against the system's MAX configuration (all slots filled).
    model_fits     = m_gb <= sys["vram_gb"]

    if not model_fits and not sys["nvlink"]:
        # Can't fit and no NVLink — cannot serve this model
        return TCOResult(
            system_name=system_name, num_nodes=0, capex_usd=0, annual_power_usd=0,
            tco_usd=0, cost_per_user=0, cost_per_mtok=None,
            predicted_tps=0, gpu_util_pct=0, fits_single_node=False,
            recommendation="Cannot fit",
            rec_reason=f"Model needs {m_gb:.0f} GB at {eff_ctx} tok; system has "
                       f"{sys['vram_gb']} GB total (no NVLink to span nodes)",
            color="#dc3545",
            mem_bw_tbs=mem_bw_tbs, vram_total_gb=sys["vram_gb"],
            gpus_per_node=full_gpus, gpus_per_copy=gpus_per_model,
            link_label=sys.get("link_label", ""), unit_price=node_price(sys, full_gpus),
            rating="Not Viable",
            rating_reason="Model does not fit available VRAM",
        )

    n_users = max(num_users, 1)

    # Effective node population. PCIe boxes (XE7745 / XE7740) are populated with the
    # FEWEST GPUs in [min_gpus, max_gpus] that serve this workload — so a job that
    # fits on 3 or 5 GPUs uses (and is billed for) exactly that many. SXM / rack
    # systems are always fully populated. Cost and power then scale with the count.
    copies_per_gpu    = max(1, int(vram_per_gpu / max(m_gb, 1e-3))) if gpus_per_model == 1 else 0
    gpus_per_node_eff = effective_node_gpus(sys, gpus_per_model, copies_per_gpu, n_users)
    vram_node_eff     = vram_per_gpu * gpus_per_node_eff

    # How many GPUs / nodes ONE model copy spans. A copy needs multiple nodes when
    # it needs more GPUs than a single (effective) node has — the slow cross-node case.
    nodes_per_copy = max(1, math.ceil(gpus_per_model / gpus_per_node_eff))

    # Interconnect efficiency — NVLink (SXM) vs PCIe spanning vs multi-node IB (CX7).
    intra_eff, inter_eff, parallel_eff = interconnect_efficiency(
        sys, gpus_per_model, nodes_per_copy)

    # Single-stream decode speed a user experiences: per-GPU memory bandwidth
    # (HBM3 > HBM2) scaled by parallel efficiency (PCIe / cross-node spanning hurts).
    per_gpu_bw   = sys["gpu_bw_gbs"] / full_gpus
    tps_per_user = gb10_tps * (per_gpu_bw / GB10_BW_GBS) * parallel_eff

    # Bandwidth contention: whole copies that co-reside on one physical GPU split
    # its memory bandwidth (see _COPY_BW_SHARE_EXP). Co-residency is the number
    # actually packed onto the busiest GPU under this load — copies_per_gpu when
    # fully loaded, fewer at light load — so a single user sees no contention.
    # gpus_per_model > 1 (a copy SPANS GPUs) → copies_per_gpu == 0 → no contention.
    coresident   = min(copies_per_gpu, n_users) if copies_per_gpu else 1
    bw_contention = coresident ** (_COPY_BW_SHARE_EXP - 1.0) if coresident > 1 else 1.0
    tps_per_user *= bw_contention

    # Concurrent model copies (1 session each) and the nodes they require.
    if nodes_per_copy > 1:
        # One copy spans several nodes → 1 session per copy; bill all nodes per copy.
        copies_per_node      = None
        provisioned_sessions = n_users
        nodes_needed         = n_users * nodes_per_copy
    else:
        vram_limited = max(1, int(vram_node_eff / max(m_gb, 1e-3)))
        if gpus_per_model > 1:
            gpu_limited     = max(1, gpus_per_node_eff // gpus_per_model)
            copies_per_node = min(vram_limited, gpu_limited)
        else:
            copies_per_node = vram_limited
        nodes_needed         = max(1, math.ceil(n_users / copies_per_node))
        provisioned_sessions = nodes_needed * copies_per_node

    gpus_total = nodes_needed * gpus_per_node_eff

    # Fleet-coordination CapEx overhead — grows with node count (see constants).
    # A single node pays none; a 200-box fleet pays the cap.
    fleet_overhead = min(_FLEET_COORD_CAP,
                         _FLEET_COORD_PER_DOUBLING * math.log2(max(nodes_needed, 1)))

    # Cost — scales with the GPUs actually populated per node (flexible PCIe boxes),
    # plus flat infra (add_infra_pct) and the node-count-scaled fleet overhead.
    capex          = (nodes_needed * node_price(sys, gpus_per_node_eff)
                      * (1.0 + add_infra_pct + fleet_overhead))
    tdp_total_w    = nodes_needed * node_tdp_w(sys, gpus_per_node_eff)
    annual_power   = tdp_total_w * 8_760 * power_rate / 1_000   # kWh
    tco            = capex + annual_power * amort_years

    # Aggregate throughput = every concurrent user served at their decode speed.
    total_tps_sys  = n_users * tps_per_user
    hourly_rate    = (capex / (amort_years * 8_760)) + (tdp_total_w * power_rate / 1_000)
    cost_per_mtok  = (hourly_rate / max(total_tps_sys * 3_600, 1)) * 1_000_000 if total_tps_sys > 0 else None
    cost_per_user  = tco / n_users
    gpu_util       = min(100.0, n_users / max(provisioned_sessions, 1) * 100.0)

    # Recommendation reason (rating supersedes this; detail cards still show it)
    fits_single  = gpus_per_model <= gpus_per_node_eff
    mem_fill_pct = (m_gb / vram_node_eff) * 100
    if nodes_per_copy > 1:
        rec = "Multi-node"
        reason = (f"1 copy spans {gpus_per_model} GPUs across {nodes_per_copy} nodes over "
                  f"{sys.get('net','IB')} — cross-node penalty (parallel eff {parallel_eff:.0%})")
    elif gpus_per_model > 1:
        rec = "Tensor-parallel"
        reason = (f"Copy spans {gpus_per_model} GPUs via {sys.get('link_label','NVLink')} "
                  f"(parallel eff {parallel_eff:.0%})")
    elif mem_fill_pct > 85:
        rec = "Tight Fit"
        reason = f"Model uses {mem_fill_pct:.0f}% of per-GPU VRAM — limited KV headroom"
    else:
        rec = "Single-GPU"
        _pop = (f"{gpus_per_node_eff}-GPU node · " if sys.get("flexible_gpus") else "")
        reason = f"{_pop}1 GPU per copy · {copies_per_node} copies/node · {sys.get('link_label','')}"
    if bw_contention < 0.999:
        reason += (f" · {coresident} copies share 1 GPU's BW "
                   f"(−{(1 - bw_contention) * 100:.0f}% tok/s/user)")
    if fleet_overhead >= 0.10:
        reason += f" · {nodes_needed} nodes → +{fleet_overhead * 100:.0f}% fleet/coord CapEx"

    return TCOResult(
        system_name     = system_name,
        num_nodes       = nodes_needed,
        capex_usd       = capex,
        annual_power_usd= annual_power,
        tco_usd         = tco,
        cost_per_user   = cost_per_user,
        cost_per_mtok   = cost_per_mtok,
        predicted_tps   = total_tps_sys,
        gpu_util_pct    = gpu_util,
        fits_single_node= fits_single,
        recommendation  = rec,
        rec_reason      = reason,
        color           = sys["color"],
        mem_bw_tbs      = mem_bw_tbs,
        vram_total_gb   = vram_node_eff,
        tps_per_user    = tps_per_user,
        bw_contention   = bw_contention,
        fleet_overhead_pct = fleet_overhead,
        gpus_per_node   = gpus_per_node_eff,
        gpus_per_copy   = gpus_per_model,
        nodes_per_copy  = nodes_per_copy,
        gpus_total      = gpus_total,
        link_label      = sys.get("link_label", ""),
        parallel_eff    = parallel_eff,
        unit_price      = node_price(sys, gpus_per_node_eff),
    )


# A system whose session utilization is below this is "Over Sized" — it provides
# more than ~2× the capacity the workload uses, i.e. more system than needed.
OVERSIZED_UTIL_PCT = 50.0


# FinTech bandwidth-bound profile: how the Best/Better/Good score is split between
# raw per-GPU memory bandwidth (the quant/HFT-relevant ceiling — LOB scatter/gather,
# Black-Scholes, risk reductions are memory-bound, not token-cost-bound) and VRAM
# capacity (can the order book / pricing grid / Monte-Carlo paths fit on a GPU).
_FINTECH_BW_WEIGHT  = 0.7
_FINTECH_CAP_WEIGHT = 0.3


def assign_ratings(results: list[TCOResult], profile: str = "llm") -> list[TCOResult]:
    """Assign a combined performance tier across a peer group of results.

    Two scoring profiles:

    * ``"llm"`` (default) — LLM inference. Performance is per-user decode speed
      (memory-bandwidth driven, → tok/s) blended 50/50 with cost efficiency
      ($/MTok over the 3-yr TCO).
    * ``"fintech"`` — bandwidth-bound quant / HPC (LOB, Black-Scholes, risk).
      These are memory-bound, not token-cost-bound, so the score is driven by
      raw per-GPU **memory bandwidth** (``_FINTECH_BW_WEIGHT``) blended with VRAM
      **capacity** (``_FINTECH_CAP_WEIGHT``); cost does not enter the score.

    Either way the top scorer is **Best** and the rest are graded relative to it;
    systems that cannot serve the workload are **Not Viable**.

    Tiers: Best · Better (≥75% of best) · Good (≥50%) · Viable · Not Viable.
    Mutates and returns the same list.
    """
    fitting = [r for r in results
               if r.recommendation != "Cannot fit" and r.predicted_tps > 0]

    if fitting and profile == "fintech":
        # Memory-bandwidth + capacity, cost-independent. Bandwidth is the FinTech
        # ceiling; capacity gates whether the working set (order book / pricing grid /
        # MC paths) fits. Both are PER-GPU properties of the part — using per-GPU VRAM
        # (not node-total) keeps a big multi-GPU rack from "winning" capacity purely by
        # GPU count, so right-sized platforms can still earn Best.
        def _per_gpu_vram(r: TCOResult) -> float:
            return r.vram_total_gb / max(r.gpus_per_node, 1)
        max_bw  = max(r.mem_bw_tbs for r in fitting) or 1.0
        max_cap = max(_per_gpu_vram(r) for r in fitting) or 1.0
        for r in fitting:
            bw_norm  = r.mem_bw_tbs / max_bw if max_bw else 0.0
            cap_norm = _per_gpu_vram(r) / max_cap if max_cap else 0.0
            r.perf_score = round(_FINTECH_BW_WEIGHT * bw_norm
                                 + _FINTECH_CAP_WEIGHT * cap_norm, 4)
        top = max((r.perf_score for r in fitting), default=0.0) or 1.0
        for r in fitting:
            ratio = r.perf_score / top
            r.rating = ("Best"   if r.perf_score >= top
                        else "Better" if ratio >= 0.75
                        else "Good"   if ratio >= 0.50
                        else "Viable")
            r.rating_reason = (
                f"{r.mem_bw_tbs:.2f} TB/s/GPU mem BW · "
                f"{_per_gpu_vram(r):,.0f} GB VRAM/GPU · "
                f"score {r.perf_score:.2f}"
            )

    elif fitting:
        # Performance = per-user decode speed (memory-bandwidth driven).
        # Cost = $/MTok efficiency (blends TCO + aggregate throughput).
        max_speed = max(r.tps_per_user for r in fitting) or 1.0
        mtoks     = [r.cost_per_mtok for r in fitting if r.cost_per_mtok]
        best_mtok = min(mtoks) if mtoks else 1.0
        for r in fitting:
            perf_norm = r.tps_per_user / max_speed if max_speed else 0.0
            cost_norm = (best_mtok / r.cost_per_mtok) if r.cost_per_mtok else 0.0
            r.perf_score = round(0.5 * perf_norm + 0.5 * cost_norm, 4)
        top = max((r.perf_score for r in fitting), default=0.0) or 1.0
        for r in fitting:
            ratio = r.perf_score / top
            r.rating = ("Best"   if r.perf_score >= top
                        else "Better" if ratio >= 0.75
                        else "Good"   if ratio >= 0.50
                        else "Viable")
            r.rating_reason = (
                f"{r.tps_per_user:,.0f} tok/s/user · "
                f"{('$%.4f/MTok' % r.cost_per_mtok) if r.cost_per_mtok else 'n/a'} · "
                f"score {r.perf_score:.2f}"
            )

    if fitting:

        # Over-provisioned: fits and performs, but uses far less than the capacity
        # it provides (low session utilization) → more system than the workload
        # needs. Flag as "Over Sized" (shown in orange) over the per-tier label.
        for r in fitting:
            if r.gpu_util_pct < OVERSIZED_UTIL_PCT:
                r.rating = "Over Sized"
                r.rating_reason = (
                    f"Over-provisioned — only {r.gpu_util_pct:.0f}% of provisioned "
                    f"sessions used; more capacity than this workload needs"
                )

    for r in results:
        if r.recommendation == "Cannot fit" or r.predicted_tps <= 0:
            r.rating = "Not Viable"
            r.rating_reason = r.rec_reason or "Does not fit / no throughput"
    return results


# Rating tier → display colour (best = green, down to red for not viable)
RATING_COLORS: dict[str, str] = {
    "Best":       "#1e8e3e",
    "Better":     "#34a853",
    "Good":       "#007DB8",
    "Viable":     "#6c757d",
    "Over Sized": "#f57c00",   # orange — fits & performs, but over-provisioned
    "Not Viable": "#dc3545",
}
# Order for sorting best → worst
RATING_ORDER: dict[str, int] = {
    "Best": 0, "Better": 1, "Good": 2, "Viable": 3, "Over Sized": 4, "Not Viable": 9,
}


def best_fit_systems(
    model_name: str, precision: str, num_users: int,
    gb10_tps: float, output_toks: int = 256, context_len: int = 512,
    amort_years: int = 3, power_rate: float = 0.12,
) -> list[TCOResult]:
    """Evaluate all systems, assign combined ratings, and sort best → worst."""
    results = []
    for name in DELL_SYSTEMS:
        try:
            r = calculate_tco(name, model_name, precision, num_users,
                              gb10_tps, output_toks, context_len,
                              amort_years, power_rate)
            results.append(r)
        except Exception:
            pass
    assign_ratings(results)
    results.sort(key=lambda r: (RATING_ORDER.get(r.rating, 5), -r.perf_score, r.tco_usd))
    return results


def calculate_tco_montecarlo(
    system_name:          str,
    resident_paths:       float,
    bytes_per_path:       int,
    timesteps:            int,
    target_paths_per_sec: float,
    amort_years:          int   = 3,
    power_rate:           float = 0.12,
    add_infra_pct:        float = 0.15,
) -> TCOResult:
    """Size a system for a memory-bound Monte-Carlo workload (bandwidth-bound quant).

    Working set (``resident_paths × bytes_per_path``) must FIT in VRAM; the target
    throughput demands memory bandwidth = ``paths/sec × bytes/path × timesteps``.
    GPUs needed = ``max(capacity, bandwidth)``. Paths are embarrassingly parallel,
    so they shard across GPUs/nodes (no NVLink requirement). The Best/Better rating
    is still the cost-independent BW+capacity score (``assign_ratings`` ``"fintech"``).
    """
    sys = DELL_SYSTEMS.get(system_name)
    if not sys:
        raise ValueError(f"Unknown system: {system_name}")

    full_gpus    = max(sys["gpus_per_node"], 1)
    vram_per_gpu = sys["vram_gb"] / full_gpus
    per_gpu_bw   = sys["gpu_bw_gbs"] / full_gpus           # GB/s
    mem_bw_tbs   = per_gpu_bw / 1000.0

    working_set_gb       = resident_paths * bytes_per_path / 1e9
    bytes_per_path_total = bytes_per_path * max(timesteps, 1)        # traffic to complete one path
    demanded_gbps        = target_paths_per_sec * bytes_per_path_total / 1e9

    # GPUs to HOLD the resident state, and to MEET the throughput — take the max.
    gpus_cap = math.ceil(working_set_gb / max(vram_per_gpu, 1e-6))
    gpus_bw  = math.ceil(demanded_gbps  / max(per_gpu_bw * MC_BW_EFF, 1e-6))
    gpus_needed = max(1, gpus_cap, gpus_bw)

    # Node packing. Fixed (SXM/rack) systems buy whole nodes; flexible PCIe boxes
    # populate exactly the GPUs needed (chassis + per-GPU).
    if sys.get("flexible_gpus"):
        node_cap     = sys.get("max_gpus", full_gpus)
        nodes_needed = max(1, math.ceil(gpus_needed / node_cap))
        gpus_total   = max(gpus_needed,
                           sys.get("min_gpus", 2) if nodes_needed == 1 else gpus_needed)
        capex_base   = nodes_needed * sys["chassis_price"] + gpus_total * sys["gpu_price"]
        tdp_w        = nodes_needed * sys["chassis_tdp_w"] + gpus_total * sys.get("gpu_tdp_w", 0)
        gpus_per_node_eff = math.ceil(gpus_total / nodes_needed)
    else:
        node_cap     = full_gpus
        nodes_needed = max(1, math.ceil(gpus_needed / node_cap))
        gpus_total   = nodes_needed * full_gpus
        capex_base   = nodes_needed * sys["system_price"]
        tdp_w        = nodes_needed * sys["system_tdp_w"]
        gpus_per_node_eff = full_gpus

    fleet_overhead = min(_FLEET_COORD_CAP,
                         _FLEET_COORD_PER_DOUBLING * math.log2(max(nodes_needed, 1)))
    capex        = capex_base * (1.0 + add_infra_pct + fleet_overhead)
    annual_power = tdp_w * 8_760 * power_rate / 1_000
    tco          = capex + annual_power * amort_years

    # Achievable throughput from the provisioned GPUs (≥ target; capacity may overshoot).
    achievable_pps  = (per_gpu_bw * MC_BW_EFF * gpus_total * 1e9) / max(bytes_per_path_total, 1)
    amort_seconds   = amort_years * 365 * 24 * 3_600
    total_bpaths    = achievable_pps * amort_seconds / 1e9
    cost_per_bpaths = tco / total_bpaths if total_bpaths > 0 else None
    vram_node_eff   = vram_per_gpu * gpus_per_node_eff

    bound  = "bandwidth" if gpus_bw >= gpus_cap else "capacity"
    reason = (f"{gpus_needed:,} GPU(s) to hit {target_paths_per_sec/1e6:,.0f}M paths/s "
              f"({bound}-bound) · {working_set_gb:,.1f} GB working set · "
              f"{per_gpu_bw * MC_BW_EFF / 1000:,.2f} TB/s usable BW/GPU")

    return TCOResult(
        system_name      = system_name,
        num_nodes        = nodes_needed,
        capex_usd        = capex,
        annual_power_usd = annual_power,
        tco_usd          = tco,
        cost_per_user    = tco / max(gpus_total, 1),       # $/GPU as a sizing handle
        cost_per_mtok    = None,
        predicted_tps    = achievable_pps,                 # >0 keeps it out of "Not Viable"
        gpu_util_pct     = 100.0,
        fits_single_node = nodes_needed == 1,
        recommendation   = "Monte-Carlo",
        rec_reason       = reason,
        color            = sys["color"],
        mem_bw_tbs       = mem_bw_tbs,
        vram_total_gb    = vram_node_eff,
        tps_per_user     = achievable_pps / max(gpus_total, 1),
        gpus_per_node    = gpus_per_node_eff,
        gpus_per_copy    = 1,
        nodes_per_copy   = 1,
        gpus_total       = gpus_total,
        link_label       = sys.get("link_label", ""),
        unit_price       = node_price(sys, gpus_per_node_eff),
        paths_per_sec    = achievable_pps,
        cost_per_bpaths  = cost_per_bpaths,
        working_set_gb   = working_set_gb,
    )


def best_fit_montecarlo(
    resident_paths:       float,
    bytes_per_path:       int,
    timesteps:            int,
    target_paths_per_sec: float,
    amort_years:          int   = 3,
    power_rate:           float = 0.12,
    add_infra_pct:        float = 0.15,
    systems:              Optional[list[str]] = None,
) -> list[TCOResult]:
    """Size every system for the MC workload, rate (BW+capacity), and sort by
    fewest GPUs then lowest TCO (the sizing answer for bandwidth-bound quant)."""
    results = []
    for name in (systems or list(DELL_SYSTEMS)):
        try:
            results.append(calculate_tco_montecarlo(
                name, resident_paths, bytes_per_path, timesteps,
                target_paths_per_sec, amort_years, power_rate, add_infra_pct))
        except Exception:
            pass
    assign_ratings(results, profile="fintech")
    results.sort(key=lambda r: (RATING_ORDER.get(r.rating, 5), r.gpus_total, r.tco_usd))
    return results


def format_usd(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:.0f}"
