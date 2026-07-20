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
        # Server Edition figures (verified 2026-07-16, Lenovo Press lp2263 + NVIDIA).
        # The catalog previously carried 1,800 GB/s / 300 W — those are the *Workstation*
        # (1,792) and *Max-Q* (300 W) numbers. Same GB202 silicon, different power/clocks:
        #   Server Edition   passive, configurable up to 600 W, 1,597 GB/s
        #   Max-Q            blower, 300 W  ← if Dell ships XE7740/R7715 as Max-Q, use 300
        "arch": "Blackwell", "class": "PCIe DW server GPU",
        "mem_gb": 96, "mem_type": "GDDR7", "mem_bw_gbs": 1_579,
        "link": "PCIe Gen5 x16", "nvlink_gbs": 0, "gpu_tdp_w": 600,
        "rt_tflops": 380,  # RT Core ray-tracing throughput (not a tensor precision)
        "tflops": {
            # Matrix headlines (FP4 4 / FP8 2 / FP16 1 PFLOPS) are with-sparsity; dense is the 2:1 half.
            "FP4":  _tf(2_000, 4_000), "FP8":  _tf(1_000, 2_000),
            "FP16": _tf(500, 1_000),   "BF16": _tf(500, 1_000),
            "FP32": _tf(120), "FP64": _tf(1.9),   # FP64 estimated, not published
        },
        "notes": "GB202 · 96 GB GDDR7 @ 1,579 GB/s (512-bit) · x8 stack = 768 GB / 12.632 TB/s · "
                 "PCIe Gen5 x16 · passive, up to 600 W. "
                 "NO NVLink — verified: NVIDIA removed NVLink from the whole RTX PRO line (RTX A6000 had "
                 "bridges, Ada dropped them, Blackwell did not bring them back), so multi-GPU is PCIe only. "
                 "FP32 125 TFLOPS · FP64 1.97 TFLOPS · RT Core 380 TFLOPS · FP4 4,000 TOPS (w/ sparsity).",
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
    "L40S": {
        "arch": "Ada Lovelace", "class": "PCIe dual-slot passive",
        "mem_gb": 48, "mem_type": "GDDR6 ECC", "mem_bw_gbs": 864,
        "link": "PCIe Gen4 x16 (64 GB/s bidirectional)", "nvlink_gbs": 0, "gpu_tdp_w": 350,
        "rt_tflops": 209,
        "tflops": {
            # NVIDIA datasheet headline numbers are "with sparsity"; dense is the 2:1 half.
            "FP8":  _tf(733, 1_466), "FP16": _tf(362.05, 733),
            "BF16": _tf(362.05, 733), "TF32": _tf(183, 366),
            "FP32": _tf(91.6), "FP64": _tf(1.43),   # FP64 estimated, not published
        },
        "notes": "18,176 CUDA · 142 RT · 568 Tensor cores · 48 GB GDDR6 ECC @ 864 GB/s · 350 W · "
                 "x8 stack = 384 GB / 6.912 TB/s · PCIe Gen4 x16 · NO NVLink — multi-GPU over PCIe only.",
    },
    "L4": {
        "arch": "Ada Lovelace", "class": "PCIe single-slot low-profile",
        "mem_gb": 24, "mem_type": "GDDR6", "mem_bw_gbs": 300,
        "link": "PCIe Gen4 x16 (64 GB/s bidirectional)", "nvlink_gbs": 0, "gpu_tdp_w": 72,
        "tflops": {
            # NVIDIA quotes these with sparsity; dense is the 2:1 half.
            "FP8":  _tf(242, 485), "FP16": _tf(121, 242),
            "BF16": _tf(121, 242), "TF32": _tf(60, 120),
            "FP32": _tf(30.3), "FP64": _tf(0.473),   # FP64 estimated, not published
        },
        "notes": "7,424 CUDA · 232 Tensor cores · 24 GB GDDR6 @ 300 GB/s · 72 W · 1-slot low-profile · "
                 "x8 stack = 192 GB / 2.4 TB/s · PCIe Gen4 x16 · NO NVLink. Lowest-power part here.",
    },
    "A16": {
        # Modelled at CARD level (64 GB / 800 GB/s / 250 W), because the platform matrix counts
        # A16 *cards* — "R770, A16, 1, 2" means 1–2 cards, not GPUs. The card-level figures are
        # exactly 4× the per-GPU ones (4 × 16 GB, 4 × 200 GB/s, 4 × 62.5 W, 4 × 17.9 TFLOPS).
        #
        # ⚠ KNOWN OPTIMISM: the card's 64 GB is four INDEPENDENT 16 GB GPUs with no NVLink, so a
        # model cannot actually span them. The fit gate treats vram_gb as poolable and will
        # therefore say a >16 GB model "fits" an A16 when physically it cannot. The source matrix
        # flags the same caveat ("not a claim of memory pooling"). A16 is a VDI part — it is in the
        # catalog for density/VDI comparison, not LLM serving. Do not read its LLM rows as real.
        "arch": "Ampere", "class": "PCIe dual-slot passive (quad-GPU board)",
        "mem_gb": 64, "mem_type": "GDDR6 ECC", "mem_bw_gbs": 800,
        "link": "PCIe Gen4 x16 (64 GB/s bidirectional)", "nvlink_gbs": 0, "gpu_tdp_w": 250,
        "gpus_per_board": 4,       # 4 × 16 GB @ 200 GB/s each — independent, NOT pooled
        "mem_gb_per_die": 16,      # the real per-GPU limit an LLM actually sees
        "tflops": {                # card totals, per the matrix (dense / sparse published explicitly)
            "FP16": _tf(71.6, 143.6), "BF16": _tf(71.6, 143.6), "TF32": _tf(36, 72), "FP32": _tf(18),
        },
        "notes": "Quad-GPU VDI board: 4 × 16 GB GDDR6 ECC = 64 GB/card @ 800 GB/s aggregate, 250 W/card, "
                 "PCIe Gen4, NO NVLink. x6 stack = 384 GB / 4.8 TB/s. Purpose-built for high-density VDI "
                 "(≤64 sessions) — the four dies are independent and cannot pool memory for one model.",
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
        "link": "NVLink 900 GB/s + PCIe Gen5 128 GB/s", "nvlink_gbs": 900, "gpu_tdp_w": 700,
        "tflops": {
            "FP8":  _tf(1_979, 3_958), "FP16": _tf(990, 1_979),
            "BF16": _tf(990, 1_979),   "TF32": _tf(495, 990), "FP32": _tf(67),
        },
        "notes": "16,896 FP32 cores · 528 Tensor Cores. Official BW 3.35 TB/s (Dell matrix rounds to 3.0).",
    },
    "H200 SXM5": {
        "arch": "Hopper", "class": "SXM5 / HGX",
        "mem_gb": 141, "mem_type": "HBM3e", "mem_bw_gbs": 4_800,
        "link": "NVLink 900 GB/s + PCIe Gen5 x16", "nvlink_gbs": 900, "gpu_tdp_w": 700,
        "tflops": {  # same Hopper compute as H100 SXM5; the uplift is memory capacity/BW
            "FP8":  _tf(1_979, 3_958), "FP16": _tf(990, 1_979),
            "BF16": _tf(990, 1_979),   "TF32": _tf(495, 990), "FP32": _tf(67), "FP64": _tf(34),
        },
        "notes": "Same Hopper tensor throughput as H100 SXM5; Dell emphasizes the 141 GB / 4.8 TB/s memory "
                 "uplift. 700 W per GPU · x8 HGX stack = 1.128 TB / 38.4 TB/s.",
    },
    "H200 NVL": {
        # The NVL card is NOT an SXM5 at a different price: it is power-limited (600 W vs 700 W),
        # so its tensor peaks are lower. The catalog previously copied the SXM5 figures here.
        "arch": "Hopper", "class": "PCIe dual-slot (NVL)",
        "mem_gb": 141, "mem_type": "HBM3e", "mem_bw_gbs": 4_800,
        "link": "NVLink 900 GB/s bridge (2- or 4-way) + PCIe Gen5 x16", "nvlink_gbs": 900,
        "gpu_tdp_w": 600,   # up to 600 W, configurable down to 450 W
        "tflops": {
            "FP8":  _tf(1_671, 3_341), "FP16": _tf(835, 1_671),
            "BF16": _tf(835, 1_671),   "TF32": _tf(418, 835), "FP32": _tf(60), "FP64": _tf(30),
        },
        "notes": "PCIe NVL variant — 141 GB HBM3e @ 4.8 TB/s, 900 GB/s NVLink bridge in 2- or 4-GPU "
                 "islands (x4 NVL4 stack = 564 GB / 19.2 TB/s). Up to 600 W. The only bridged part in "
                 "this catalog — the sole user of the `nvlink-bridge` link class.",
    },
    # B200/B300 are split by cooling variant because the AC and PC builds genuinely differ:
    # B300 AC carries 288 GB vs PC's 270 GB, and B200 AC/PC differ slightly in bandwidth.
    # A single shared spec cannot represent both, and memory drives the model-fit gate.
    # tflops below follow the supplied matrix (FP4 18 / FP8 9 / FP16 4.5 PFLOPS are the
    # with-sparsity headlines; dense is the 2:1 half).
    "B200 AC": {
        "arch": "Blackwell", "class": "SXM / HGX (air-cooled)",
        "mem_gb": 180, "mem_type": "HBM3e", "mem_bw_gbs": 7_700,
        "link": "NVLink 5, 1.8 TB/s per GPU", "nvlink_gbs": 1_800, "gpu_tdp_w": 1_000,
        "tflops": {
            "FP4":  _tf(9_000, 18_000), "FP8":  _tf(4_500, 9_000),
            "FP16": _tf(2_250, 4_500),  "BF16": _tf(2_250, 4_500),
            "TF32": _tf(1_125, 2_250),  "FP32": _tf(75), "FP64": _tf(37),
        },
        "notes": "180 GB HBM3e @ 7.7 TB/s · 1,000 W per GPU · x8 HGX stack = 1.44 TB / 61.6 TB/s.",
    },
    "B200 PC": {
        "arch": "Blackwell", "class": "SXM / HGX (liquid-cooled)",
        "mem_gb": 180, "mem_type": "HBM3e", "mem_bw_gbs": 7_750,
        "link": "NVLink 5, 1.8 TB/s per GPU", "nvlink_gbs": 1_800, "gpu_tdp_w": 1_000,
        "tflops": {
            "FP4":  _tf(9_000, 18_000), "FP8":  _tf(4_500, 9_000),
            "FP16": _tf(2_250, 4_500),  "BF16": _tf(2_250, 4_500),
            "TF32": _tf(1_125, 2_250),  "FP32": _tf(75), "FP64": _tf(37),
        },
        "notes": "180 GB HBM3e @ 7.75 TB/s · 1,000 W per GPU · x8 HGX stack = 1.44 TB / 62 TB/s.",
    },
    "B300 AC": {
        "arch": "Blackwell Ultra", "class": "SXM / HGX (air-cooled)",
        "mem_gb": 288, "mem_type": "HBM3e", "mem_bw_gbs": 8_000,
        "link": "NVLink 5, 1.8 TB/s per GPU", "nvlink_gbs": 1_800, "gpu_tdp_w": 1_100,
        "tflops": {
            "FP4":  _tf(9_000, 18_000), "FP8":  _tf(4_500, 9_000),
            "FP16": _tf(2_250, 4_500),  "BF16": _tf(2_250, 4_500),
            "TF32": _tf(1_125, 2_250),  "FP32": _tf(75), "FP64": _tf(1.25),
        },
        "notes": "288 GB HBM3e @ 8 TB/s · 1,100 W per GPU · x8 HGX stack = 2.304 TB / 64 TB/s. "
                 "Note FP64 is only 1.25 TFLOPS — Blackwell Ultra trades FP64 for low-precision AI.",
    },
    "B300 PC": {
        "arch": "Blackwell Ultra", "class": "SXM / NVL8 (liquid-cooled)",
        "mem_gb": 270, "mem_type": "HBM3e", "mem_bw_gbs": 8_000,   # PC build carries 270 GB, not 288
        "link": "NVLink 5, 1.8 TB/s per GPU", "nvlink_gbs": 1_800, "gpu_tdp_w": 1_100,
        "tflops": {
            "FP4":  _tf(9_000, 18_000), "FP8":  _tf(4_500, 9_000),
            "FP16": _tf(2_250, 4_500),  "BF16": _tf(2_250, 4_500),
            "TF32": _tf(1_125, 2_250),  "FP32": _tf(75), "FP64": _tf(1.25),
        },
        "notes": "270 GB HBM3e @ 8 TB/s · 1,100 W per GPU · x8 NVL8 stack = 2.16 TB / 64 TB/s.",
    },
    # NVL4 is a fixed 4-GPU system whose published total is 744 GB / 32 TB/s. Deriving from the
    # generic GB200 (192 GB) would give 768 GB, so the NVL4 build gets its own per-GPU spec that
    # multiplies up to the real system totals: 4 × 186 = 744 GB, 4 × 8,000 = 32 TB/s.
    "GB200 NVL4": {
        "arch": "Grace Blackwell superchip", "class": "NVL4 fixed system",
        "mem_gb": 186, "mem_type": "HBM3e", "mem_bw_gbs": 8_000,
        "link": "NVLink 5 + NVLink-C2C 900 GB/s CPU↔GPU", "nvlink_gbs": 1_800, "gpu_tdp_w": 700,
        "tflops": {  # matrix gives NVL4 system totals; per-GPU = /4
            "FP4":  _tf(10_000, 20_000), "FP8":  _tf(5_000, 10_000),
            "FP16": _tf(2_500, 5_000),   "BF16": _tf(2_500, 5_000),
            "TF32": _tf(1_250, 2_500),   "FP32": _tf(80), "FP64": _tf(40),
        },
        "notes": "Fixed NVL4 system: 744 GB HBM3e / 32 TB/s total · FP4 80 PFLOPS · FP8 40 PFLOPS · "
                 "FP16 20 PFLOPS · FP32 320 TFLOPS · FP64 160 TFLOPS · 700 W per GPU (2.8 kW GPU complex).",
    },
    "GB200": {
        "arch": "Grace Blackwell superchip", "class": "NVL72-class",
        "mem_gb": 192, "mem_type": "HBM3E", "mem_bw_gbs": 8_000,  # per Blackwell GPU
        "link": "NVLink 5 1.8 TB/s per GPU + NVLink-C2C 900 GB/s CPU↔GPU", "nvlink_gbs": 1_800,
        "gpu_tdp_w": 700,
        "tflops": {  # per Blackwell GPU (same silicon as B200)
            "FP4":  _tf(10_000, 20_000), "FP8":  _tf(5_000, 10_000),
            "FP16": _tf(2_500, 5_000),   "BF16": _tf(2_500, 5_000),   "TF32": _tf(1_250, 2_500),
        },
        "notes": "Superchip: 372 GB HBM3E / 16 TB/s (2 GPUs). NVL72 system: 13.4 TB HBM3E / 576 TB/s; "
                 "1,440 PFLOPS FP4 (sparse) / 720 PFLOPS FP8 (sparse) across 72 GPUs.",
    },
    "GB300": {
        "arch": "Grace Blackwell Ultra superchip", "class": "NVL72-class",
        "mem_gb": 288, "mem_type": "HBM3e", "mem_bw_gbs": 8_000,  # per GPU → ×72 = 20.736 TB / 576 TB/s
        "link": "NVLink 5 / NVLink switching + ConnectX-8 800 Gb/s", "nvlink_gbs": 1_800,
        "gpu_tdp_w": 1_100,   # rack power is set explicitly on XE9712 (136 kW), not derived from this
        "tflops": {  # per Blackwell Ultra GPU — matrix NVL72 totals ÷ 72
            "FP4":  _tf(15_000, 20_000), "FP8":  _tf(5_000, 10_000),
            "FP16": _tf(2_500, 5_000),   "BF16": _tf(2_500, 5_000),   "TF32": _tf(1_250, 2_500),
        },
        "notes": "NVL72 rack totals: 20.736 TB HBM3e / 576 TB/s · FP4 1,080 PFLOPS · FP8 720 PFLOPS · "
                 "FP16 360 PFLOPS · FP32 6 PFLOPS · FP64 100 TFLOPS · 136 kW per rack.",
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

# The catalog is DATA (PLATFORMS + _CATALOG_ROWS) plus a builder, not ~75 hand-written
# dicts. Every row is a chassis plus a linear per-GPU cost/power, which is exactly what
# node_price()/node_tdp_w() already model, so there is no new pricing mechanism here.
#
# Three pricing shapes:
#   1. Flexible (base + GPUs)  — R-series, XE7745/7740, edge. price = platform + N × gpu_price
#   2. Fixed x8 (base + 8)     — XE97xx HGX. Same mechanism with min_gpus == max_gpus == 8.
#   3. Fixed, GPU-INCLUSIVE    — XE8712 NVL4, XE9712 NVL72, Dell GB10. Flat `system_price`
#                                already contains the accelerators; the per-GPU figure is a
#                                modeling reference and is NEVER added on top.
#
# Source: Dell 17G/16G platform matrix supplied 2026-07-16. Prices are simulated list-price
# estimates for internal modeling only — NOT quote-ready. 15G platforms intentionally omitted.
# GPUs off the current matrix for EOL/EOML (H100 NVL, L40, HGX H200 x4, HGX H100 x4) omitted.
#
# PRICE OUTLIERS — deliberate, do NOT "fix": near-identical boxes disagree in the source
# (XE9680L $570K vs XE9685L $125K, both B200 PC x8; XE9780 B300 $250K vs XE9785 $75K). Per the
# data owner the higher figure stands ("memory prices are climbing"); entered verbatim, unnormalized.
#
# CHASSIS POWER: per the data owner (2026-07-16), assume a 1 kW platform base, and 1.5 kW for
# 8-way servers (any platform that can hold 8 GPUs). This is the 0-GPU base only — per-GPU watts
# come from GPU_SPECS[...]["gpu_tdp_w"] and are added on top by node_tdp_w(). Rack-scale rows do
# not use the rule: they carry the published system figure (NVL4 GPU complex 2.8 kW + base;
# NVL72 136 kW per rack).

DELL_SYSTEMS: dict[str, dict] = {
    "Dell GB10": {
        "category":        "Edge / Workstation",
        "system_price":    8_000,        # GPU-inclusive (shape 3) — the superchip IS the system
        "gpus_per_node":   1,
        "gpu_spec":        None,          # Dell GB10 not in supplied spec sheet — keep explicit values
        "gpu_model":       "Dell GB10 (Grace-Blackwell)",
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
        "arch_family":     "Grace-Blackwell",
        "color":           "#2a78d6",    # = ARCH_COLORS["Grace-Blackwell"] (defined below)
        "notes":           "Current demo HW · 128 GB unified · lowest $/watt",
        "best_for":        ["LLM ≤32B FP16", "FinTech edge", "VLM batch ≤64"],
    },

    # --- Legacy: not in the 2026-07 matrix, retained non-default -------------------------
    # XE9640 (4× H100 SXM5) was DELETED — HGX H100 (x4) is EOL/EOML per the matrix notes.
    # This H200 x4 row is likewise flagged EOL in those notes but is retained by request.
    # No new pricing was supplied for XE9640, so its original figures stand.
    "Dell PowerEdge XE9640 (4× H200 SXM 141GB)": {
        "category":        "Legacy",
        "system_price":    390_000,
        "gpus_per_node":   4,
        "gpu_spec":        "H200 SXM5",  # derives VRAM / BW / TFLOPS
        "gpu_model":       "H200 SXM 141GB",
        "system_tdp_w":    6_500,        # 4 × ~850 W SXM5 + platform
        "rack_u":          2,
        "nw_gbps":         400,          # 1× ConnectX-7 — limited node-to-node scale-out
        "nvlink":          True,
        "sxm":             True,
        "gpu_link":        "nvlink4",    # SXM5 NVLink-4 — fast tensor-parallel
        "link_label":      "NVLink-4 (SXM5)",
        "net":             "1× ConnectX-7 400G IB",
        "arch_family":     "Hopper",
        "color":           "#e87ba4",    # = ARCH_COLORS["Hopper"]
        "notes":           "EOL/EOML — retained for comparison only. 4× H200 141GB SXM · HBM3e 4.8 TB/s/GPU · "
                           "564 GB VRAM · NVLink-4; single CX-7 → weak multi-node scale-out",
        "best_for":        ["LLM 70B FP16", "Large context (32K+)", "FinTech HPC"],
    },
    # GB200 NVL72 is absent from the 2026-07 matrix (which lists GB300 NVL72 via XE9712) but is
    # not EOL-flagged either — retained non-default at its original price.
    "Dell NVL72 (GB200 NVL72 Rack)": {
        "category":        "Legacy",
        "system_price":    5_200_000,    # GPU-inclusive (shape 3)
        "gpus_per_node":   72,           # atomic unit: one full 72-GPU NVLink rack
        "gpu_spec":        "GB200",      # derives VRAM / BW / TFLOPS
        "gpu_model":       "GB200 192GB",
        "system_tdp_w":    120_000,
        "rack_u":          42,           # full rack
        "nw_gbps":         25_600,
        "nvlink":          True,
        "sxm":             True,
        "gpu_link":        "nvlink-switch",  # all 72 GPUs in one NVLink switch domain
        "link_label":      "NVLink Switch (72-GPU rack)",
        "net":             "NVLink rack fabric",
        "rack_unit":       True,         # billed only in whole 72-GPU racks
        "arch_family":     "Grace-Blackwell",
        "color":           "#2a78d6",    # = ARCH_COLORS["Grace-Blackwell"]
        "notes":           "Not on the current matrix. Rack-scale unit — minimum 72 GPUs · 13.8 TB VRAM · "
                           "single NVLink domain (no slow cross-node hops within the rack)",
        "best_for":        ["405B+ models", "Hyperscale LLM training", "Enterprise AI platform"],
    },
}

# ---------------------------------------------------------------------------
# Platform (chassis) facts — one entry per platform, independent of the GPU
# ---------------------------------------------------------------------------
# chassis_tdp_w is the 0-GPU platform base (ESTIMATED — see header note).
# nw_gbps drives the cross-node penalty in interconnect_efficiency() (400 = one ConnectX-7).

_P = dict  # brevity for the table below

PLATFORMS: dict[str, dict] = {
    # --- 17G rack-scale (GPU-inclusive pricing) -------------------------------------
    "XE8712":  _P(gen="17G", cat="Rack Scale",  u=10, tdp=3_800,   nw=3_200,  net="ConnectX-8 800G",
                  desc="GB200 NVL4 — 4-GPU integrated NVLink node"),
    "XE9712":  _P(gen="17G", cat="Rack Scale",  u=42, tdp=136_000, nw=28_800, net="NVLink rack fabric + ConnectX-8 800G",
                  desc="GB300 NVL72 — 72-GPU integrated NVLink rack"),
    # --- 17G 8-GPU HGX --------------------------------------------------------------
    "XE9780":  _P(gen="17G", cat="8-GPU HGX",   u=8,  tdp=1_500, nw=6_400, net="8× ConnectX-8 800G IB", desc="8-GPU SXM air-cooled"),
    "XE9785":  _P(gen="17G", cat="8-GPU HGX",   u=8,  tdp=1_500, nw=6_400, net="8× ConnectX-8 800G IB", desc="8-GPU SXM air-cooled"),
    "XE9785L": _P(gen="17G", cat="8-GPU HGX",   u=8,  tdp=1_500, nw=6_400, net="8× ConnectX-8 800G IB", desc="8-GPU liquid-cooled"),
    "XE9780L (GNR AP)": _P(gen="17G", cat="8-GPU HGX", u=8, tdp=1_500, nw=6_400, net="8× ConnectX-8 800G IB",
                           desc="8-GPU liquid-cooled · Granite Rapids AP"),
    "XE9780L (GNR SP)": _P(gen="17G", cat="8-GPU HGX", u=8, tdp=1_500, nw=6_400, net="8× ConnectX-8 800G IB",
                           desc="8-GPU liquid-cooled · Granite Rapids SP"),
    # --- 17G flexible PCIe boxes ----------------------------------------------------
    "XE7745":  _P(gen="17G", cat="Flexible PCIe", u=4, tdp=1_500, nw=800, net="2× ConnectX-7 400G IB",
                  desc="Flexible 4U PCIe box — partially populated to fit the workload"),
    "XE7740":  _P(gen="17G", cat="Flexible PCIe", u=4, tdp=1_500, nw=800, net="2× ConnectX-7 400G IB",
                  desc="Flexible 4U PCIe box — partially populated to fit the workload"),
    # --- 17G mainstream servers -----------------------------------------------------
    "R770":    _P(gen="17G", cat="Mainstream Server", u=2, tdp=1_000, nw=200, net="2× 100GbE", desc="2U dual-socket Xeon 6"),
    "R7715":   _P(gen="17G", cat="Mainstream Server", u=2, tdp=1_000, nw=200, net="2× 100GbE", desc="2U single-socket EPYC"),
    "R7725":   _P(gen="17G", cat="Mainstream Server", u=2, tdp=1_000, nw=200, net="2× 100GbE", desc="2U dual-socket EPYC"),
    "R570":    _P(gen="17G", cat="Mainstream Server", u=2, tdp=1_000, nw=200, net="2× 25GbE",  desc="2U value platform"),
    "R470":    _P(gen="17G", cat="Mainstream Server", u=2, tdp=1_000, nw=200, net="2× 25GbE",  desc="2U value platform"),
    # --- 17G 1U density -------------------------------------------------------------
    "R670":    _P(gen="17G", cat="Density 1U", u=1, tdp=1_000, nw=100, net="2× 25GbE", desc="1U dual-socket Xeon 6"),
    "R6715":   _P(gen="17G", cat="Density 1U", u=1, tdp=1_000, nw=100, net="2× 25GbE", desc="1U single-socket EPYC"),
    "R6725":   _P(gen="17G", cat="Density 1U", u=1, tdp=1_000, nw=100, net="2× 25GbE", desc="1U dual-socket EPYC"),

    # --- 16G 8-GPU HGX --------------------------------------------------------------
    "XE9680":  _P(gen="16G", cat="8-GPU HGX", u=6, tdp=1_500, nw=3_200, net="8× ConnectX-7 400G IB", desc="8-GPU SXM air-cooled"),
    "XE9680L": _P(gen="16G", cat="8-GPU HGX", u=6, tdp=1_500, nw=3_200, net="8× ConnectX-7 400G IB", desc="8-GPU direct liquid-cooled"),
    "XE9685L": _P(gen="16G", cat="8-GPU HGX", u=6, tdp=1_500, nw=3_200, net="8× ConnectX-7 400G IB", desc="8-GPU direct liquid-cooled"),
    # --- 16G mainstream servers -----------------------------------------------------
    "R760XA":  _P(gen="16G", cat="Mainstream Server", u=2, tdp=1_500,   nw=200, net="2× 100GbE", desc="2U GPU-dense platform"),
    "R760":    _P(gen="16G", cat="Mainstream Server", u=2, tdp=1_000,   nw=200, net="2× 100GbE", desc="2U dual-socket Xeon"),
    "R760xd2": _P(gen="16G", cat="Mainstream Server", u=2, tdp=1_000,   nw=200, net="2× 25GbE",  desc="2U storage-dense platform"),
    "R7625":   _P(gen="16G", cat="Mainstream Server", u=2, tdp=1_000,   nw=200, net="2× 100GbE", desc="2U dual-socket EPYC"),
    "R7615":   _P(gen="16G", cat="Mainstream Server", u=2, tdp=1_000,   nw=200, net="2× 100GbE", desc="2U single-socket EPYC"),
    "R960":    _P(gen="16G", cat="Mainstream Server", u=4, tdp=1_000, nw=200, net="2× 100GbE", desc="4U four-socket Xeon"),
    # --- 16G 1U density -------------------------------------------------------------
    "R660":    _P(gen="16G", cat="Density 1U", u=1, tdp=1_000, nw=100, net="2× 25GbE", desc="1U dual-socket Xeon"),
    "R6615":   _P(gen="16G", cat="Density 1U", u=1, tdp=1_000, nw=100, net="2× 25GbE", desc="1U single-socket EPYC"),
    "R6625":   _P(gen="16G", cat="Density 1U", u=1, tdp=1_000, nw=100, net="2× 25GbE", desc="1U dual-socket EPYC"),
    # --- 16G edge / tower -----------------------------------------------------------
    "T560":    _P(gen="16G", cat="Edge / Rugged", u=5, tdp=1_000, nw=50, net="2× 25GbE", desc="Tower platform"),
    "XR7620":  _P(gen="16G", cat="Edge / Rugged", u=2, tdp=1_000,   nw=50, net="2× 25GbE", desc="Short-depth rugged edge"),
    "XR5610":  _P(gen="16G", cat="Edge / Rugged", u=1, tdp=1_000,   nw=50, net="2× 25GbE", desc="Short-depth rugged edge"),
    "XR8620t": _P(gen="16G", cat="Edge / Rugged", u=2, tdp=1_000,   nw=50, net="2× 25GbE", desc="Telecom rugged edge"),
}

# ---------------------------------------------------------------------------
# Catalog rows: (platform, gpu_spec, gpu_label, min_gpus, max_gpus, platform_price, gpu_price)
# ---------------------------------------------------------------------------
# gpu_label is the matrix's own "Current NVIDIA GPU model" text — it keeps keys faithful to the
# source AND unique per platform (XE9780 B200 AC vs B300 AC). gpu_spec must exist in GPU_SPECS;
# _validate_catalog() enforces that at import.

_CATALOG_ROWS: list[tuple] = [
    # ---- 17G rack-scale — GPU-INCLUSIVE: gpu_price is a reference, never added -------
    ("XE8712",  "GB200 NVL4", "GB200 NVL4", 4, 4,   993_000, 248_250),
    ("XE9712",  "GB300", "GB300 NVL72",  72, 72, 6_200_000,  86_111),
    # ---- 17G 8-GPU HGX (base + 8 × GPU) ---------------------------------------------
    ("XE9785",  "B300 AC", "B300 AC (x8)", 8,  8,    75_000,  53_000),   # ⚠ outlier vs XE9780 B300 $250K
    ("XE9780",  "B200 AC", "B200 AC (x8)", 8,  8,   250_000,  47_500),
    ("XE9780",  "B300 AC", "B300 AC (x8)", 8,  8,   250_000,  53_000),
    ("XE9785L", "B300 PC", "B300 PC (x8)", 8,  8,   150_000,  53_000),
    ("XE9785L", "B200 PC", "B200 PC (x8)", 8,  8,   150_000,  47_500),
    ("XE9780L (GNR AP)", "B300 PC", "B300 PC (x8)", 8, 8, 200_000, 53_000),
    ("XE9780L (GNR SP)", "B300 PC", "B300 PC (x8)", 8, 8, 185_000, 53_000),
    ("XE9780L (GNR SP)", "B200 PC", "B200 PC (x8)", 8, 8, 185_000, 47_500),
    # ---- 17G flexible PCIe ----------------------------------------------------------
    ("XE7745", "RTX PRO 6000 BSE", "RTX Pro 6000 BSE", 1, 8, 35_000, 13_500),
    ("XE7745", "RTX PRO 4500 BSE", "RTX Pro 4500 BSE", 1, 8, 35_000,  3_800),
    ("XE7745", "H200 NVL",         "H200 NVL",         1, 8, 35_000, 34_500),
    ("XE7745", "L40S",             "L40S",             1, 8, 35_000,  9_971),
    ("XE7745", "L4",               "L4",               1, 6, 35_000,  4_200),
    ("XE7740", "RTX PRO 6000 BSE", "RTX Pro 6000 BSE", 1, 8, 45_000, 13_500),
    ("XE7740", "RTX PRO 4500 BSE", "RTX Pro 4500 BSE", 1, 8, 45_000,  3_800),
    ("XE7740", "H200 NVL",         "H200 NVL",         1, 8, 45_000, 34_500),
    ("XE7740", "L40S",             "L40S",             1, 8, 45_000,  9_971),
    ("XE7740", "L4",               "L4",               1, 6, 45_000,  4_200),
    # ---- 17G mainstream -------------------------------------------------------------
    ("R770",  "RTX PRO 6000 BSE", "RTX Pro 6000 BSE", 1, 2, 25_000, 13_500),
    ("R770",  "RTX PRO 4500 BSE", "RTX Pro 4500 BSE", 1, 2, 25_000,  3_800),
    ("R770",  "H200 NVL",         "H200 NVL",         1, 2, 25_000, 34_500),
    ("R770",  "L40S",             "L40S",             1, 2, 25_000,  9_971),
    ("R770",  "L4",               "L4",               1, 6, 25_000,  4_200),
    ("R770",  "A16",              "A16",              1, 2, 25_000,  8_000),
    ("R7725", "RTX PRO 6000 BSE", "RTX Pro 6000 BSE", 1, 2, 19_000, 13_500),
    ("R7725", "RTX PRO 4500 BSE", "RTX Pro 4500 BSE", 1, 3, 19_000,  3_800),
    ("R7725", "H200 NVL",         "H200 NVL",         1, 2, 19_000, 34_500),
    ("R7725", "L40S",             "L40S",             1, 2, 19_000,  9_971),
    ("R7725", "L4",               "L4",               1, 6, 19_000,  4_200),
    ("R7725", "A16",              "A16",              1, 2, 19_000,  8_000),
    ("R7715", "RTX PRO 6000 BSE", "RTX Pro 6000 BSE", 1, 3, 27_500, 13_500),
    ("R7715", "RTX PRO 4500 BSE", "RTX Pro 4500 BSE", 1, 3, 27_500,  3_800),
    ("R7715", "H200 NVL",         "H200 NVL",         1, 3, 27_500, 34_500),
    ("R7715", "L40S",             "L40S",             1, 3, 27_500,  9_971),
    ("R7715", "L4",               "L4",               1, 6, 27_500,  4_200),
    ("R7715", "A16",              "A16",              1, 3, 27_500,  8_000),
    ("R570",  "RTX PRO 4500 BSE", "RTX Pro 4500 BSE", 1, 3, 11_600,  3_800),
    ("R570",  "L40S",             "L40S",             1, 3, 11_600,  9_971),
    ("R570",  "L4",               "L4",               1, 4, 11_600,  4_200),
    ("R470",  "L4",               "L4",               1, 4, 12_000,  4_200),
    # ---- 17G 1U density -------------------------------------------------------------
    ("R670",  "L4", "L4", 1, 3, 16_000, 4_200),
    ("R6725", "L4", "L4", 1, 3, 19_400, 4_200),
    ("R6715", "L4", "L4", 1, 3, 14_000, 4_200),

    # ---- 16G 8-GPU HGX --------------------------------------------------------------
    ("XE9685L", "B200 PC",   "B200 PC (x8)",       8, 8, 125_000, 47_500),   # ⚠ outlier vs XE9680L $570K
    ("XE9685L", "H200 SXM5", "H200 SXM (x8) DLC",  8, 8, 125_000, 46_000),
    ("XE9680L", "B200 PC",   "B200 PC (x8)",       8, 8, 570_000, 47_500),   # ⚠ outlier — kept per data owner
    ("XE9680L", "H200 SXM5", "H200 SXM (x8) DLC",  8, 8, 570_000, 46_000),
    ("XE9680",  "H200 SXM5", "H200 SXM5 (x8) AC",  8, 8,  65_000, 46_000),
    # ---- 16G mainstream -------------------------------------------------------------
    ("R760XA",  "L40S", "L40S", 2, 4, 40_000, 9_971),
    ("R760XA",  "L4",   "L4",   2, 8, 40_000, 4_200),
    ("R760XA",  "A16",  "A16",  2, 4, 40_000, 8_000),
    ("R760",    "L40S", "L40S", 1, 2, 16_100, 9_971),
    ("R760",    "L4",   "L4",   1, 4, 16_100, 4_200),
    ("R760",    "A16",  "A16",  1, 2, 16_100, 8_000),
    ("R760xd2", "L4",   "L4",   1, 2, 12_000, 4_200),
    ("R7625",   "L40S", "L40S", 1, 2, 16_000, 9_971),
    ("R7625",   "L4",   "L4",   1, 4, 16_000, 4_200),
    ("R7625",   "A16",  "A16",  1, 2, 16_000, 8_000),
    ("R7615",   "L40S", "L40S", 1, 3, 13_000, 9_971),
    ("R7615",   "L4",   "L4",   1, 4, 13_000, 4_200),
    ("R7615",   "A16",  "A16",  1, 3, 13_000, 8_000),
    ("R960",    "L40S", "L40S", 1, 4, 38_000, 9_971),
    ("R960",    "A16",  "A16",  1, 4, 38_000, 8_000),
    # ---- 16G 1U density -------------------------------------------------------------
    ("R660",  "L4", "L4", 1, 2, 12_000, 4_200),
    ("R6615", "L4", "L4", 1, 3, 13_000, 4_200),
    ("R6625", "L4", "L4", 1, 3, 18_200, 4_200),
    # ---- 16G edge / tower -----------------------------------------------------------
    ("T560",    "L4",   "L4",   1, 5, 13_000, 4_200),
    ("XR7620",  "L40S", "L40S", 1, 2, 30_000, 9_971),
    ("XR7620",  "L4",   "L4",   1, 5, 30_000, 4_200),
    ("XR5610",  "L4",   "L4",   1, 2, 22_900, 4_200),
    ("XR8620t", "L4",   "L4",   1, 3, 17_000, 4_200),
]

# Platforms whose Platform Price is the INTEGRATED solution estimate (shape 3): the accelerators
# are already in the price, so gpu_price is recorded as a reference and never added.
_RACK_SCALE_PLATFORMS = {"XE8712", "XE9712"}

# GPU → (gpu_link class, nvlink?, sxm?, link_label). Verified 2026-07-16:
#   * the entire RTX PRO line has NO NVLink — multi-GPU is PCIe Gen5 only
#   * L40S / L4 / A16 have no NVLink either
#   * H200 NVL is the ONLY bridged part here — it makes `nvlink-bridge` live for the first time
_GPU_LINK: dict[str, tuple[str, bool, bool, str]] = {
    "RTX PRO 6000 BSE": ("pcie",          False, False, "PCIe Gen5"),
    "RTX PRO 4500 BSE": ("pcie",          False, False, "PCIe Gen5"),
    "L40S":             ("pcie",          False, False, "PCIe Gen4"),
    "L4":               ("pcie",          False, False, "PCIe Gen4"),
    "A16":              ("pcie",          False, False, "PCIe Gen4"),
    "H200 NVL":         ("nvlink-bridge", True,  False, "NVLink bridge (2–4 way)"),
    "H200 SXM5":        ("nvlink4",       True,  True,  "NVLink-4 (SXM5)"),
    "H100 SXM5":        ("nvlink4",       True,  True,  "NVLink-4 (SXM5)"),
    "B200 AC":          ("nvlink5",       True,  True,  "NVLink-5 (SXM)"),
    "B200 PC":          ("nvlink5",       True,  True,  "NVLink-5 (SXM)"),
    "B300 AC":          ("nvlink5",       True,  True,  "NVLink-5 (SXM)"),
    "B300 PC":          ("nvlink5",       True,  True,  "NVLink-5 (NVL8)"),
    "B200":             ("nvlink5",       True,  True,  "NVLink-5 (SXM)"),
    "B300":             ("nvlink5",       True,  True,  "NVLink-5 (SXM)"),
    "GB200 NVL4":       ("nvlink-switch", True,  True,  "NVLink (fixed NVL4)"),
    "GB200":            ("nvlink-switch", True,  True,  "NVLink Switch (rack)"),
    "GB300":            ("nvlink-switch", True,  True,  "NVLink Switch (rack)"),
}

# Chart colors are assigned by ARCHITECTURE FAMILY, not per system: at ~75 systems a per-system
# hue is meaningless, and a categorical palette must never exceed ~8 hues or be cycled. These six
# are the validated default categorical palette's first six slots, in fixed order (validated
# light-surface: worst adjacent CVD ΔE 9.1, normal-vision ΔE 19.6 — all checks pass). Three sit
# below 3:1 contrast, which obliges "relief": the charts carry direct labels + a table view.
_ARCH_FAMILY: dict[str, str] = {
    "RTX PRO 6000 BSE": "Blackwell", "RTX PRO 4500 BSE": "Blackwell",
    "B200": "Blackwell", "B300": "Blackwell",
    "B200 AC": "Blackwell", "B200 PC": "Blackwell",
    "B300 AC": "Blackwell", "B300 PC": "Blackwell",
    "GB200 NVL4": "Grace-Blackwell",
    "GB200": "Grace-Blackwell", "GB300": "Grace-Blackwell",
    "H100 SXM5": "Hopper", "H100 PCIe": "Hopper", "H100 NVL": "Hopper",
    "H200 SXM5": "Hopper", "H200 NVL": "Hopper",
    "L40S": "Ada Lovelace", "L4": "Ada Lovelace",
    "A16": "Ampere",
    "Vera Rubin / VR200": "Rubin",
}
ARCH_COLORS: dict[str, str] = {
    "Grace-Blackwell": "#2a78d6",   # slot 1 blue   — the GB10 family
    "Blackwell":       "#008300",   # slot 2 green
    "Hopper":          "#e87ba4",   # slot 3 magenta
    "Ada Lovelace":    "#eda100",   # slot 4 yellow
    "Ampere":          "#1baf7a",   # slot 5 aqua
    "Rubin":           "#eb6834",   # slot 6 orange
}
_FALLBACK_COLOR = "#898781"  # muted ink — an unmapped family is visibly neutral, never a new hue


def _build_catalog() -> None:
    """Expand PLATFORMS × _CATALOG_ROWS into DELL_SYSTEMS entries.

    Everything derivable is derived — `system_price`, `system_tdp_w` and `gpu_tdp_w` come from
    the row + GPU_SPECS rather than being hand-carried, because hand-synced duplicates of those
    had already drifted in the previous literal catalog.
    """
    for platform, gpu_spec, gpu_label, min_g, max_g, plat_price, gpu_price in _CATALOG_ROWS:
        p    = PLATFORMS[platform]
        spec = GPU_SPECS[gpu_spec]
        link, nvlink, sxm, link_label = _GPU_LINK[gpu_spec]
        family   = _ARCH_FAMILY.get(gpu_spec, "")
        rack     = platform in _RACK_SCALE_PLATFORMS
        gpu_tdp  = spec.get("gpu_tdp_w", 0)
        key      = f"Dell PowerEdge {platform} ({gpu_label})"

        entry: dict = {
            "category":      p["cat"],
            "generation":    p["gen"],
            "platform":      platform,      # chassis alone — chart labels use this, never string-parsing
            "gpu_label":     gpu_label,     # the matrix's own GPU text — makes each row uniquely labelable
            "gpus_per_node": max_g,
            "gpu_spec":      gpu_spec,
            "gpu_model":     f"{gpu_label} · {spec['mem_gb']} GB {spec['mem_type']}",
            "rack_u":        p["u"],
            "nw_gbps":       p["nw"],
            "nvlink":        nvlink,
            "sxm":           sxm,
            "gpu_link":      link,
            "link_label":    link_label,
            "net":           p["net"],
            "arch_family":   family,
            "color":         ARCH_COLORS.get(family, _FALLBACK_COLOR),
            "gpu_price_ref": gpu_price,   # always recorded; only ADDED for non-rack-scale shapes
            "best_for":      [],
        }

        if rack:
            # Shape 3 — GPU-inclusive. Flat price; gpu_price is NEVER added (node_price() returns
            # system_price unchanged for a non-flexible entry, same as Dell GB10).
            entry.update({
                "system_price": plat_price,
                "system_tdp_w": p["tdp"],
                "rack_unit":    True,
                "notes":        f"{p['desc']} · integrated solution price (GPUs included; the per-GPU figure "
                                f"is a modeling reference only) · {max_g}× {gpu_label}",
            })
        else:
            # Shapes 1 & 2 — base + GPUs. A fixed x8 box is just min_gpus == max_gpus == 8.
            entry.update({
                "flexible_gpus": True,
                "min_gpus":      min_g,
                "max_gpus":      max_g,
                "chassis_price": plat_price,
                "gpu_price":     gpu_price,
                "chassis_tdp_w": p["tdp"],
                "gpu_tdp_w":     gpu_tdp,
                # Derived, not authored — full-config reference used by the UI price table.
                "system_price":  plat_price + max_g * gpu_price,
                "system_tdp_w":  p["tdp"] + max_g * gpu_tdp,
                "notes":         f"{p['desc']} · {min_g}–{max_g}× {gpu_label} · {spec['mem_gb']} GB "
                                 f"{spec['mem_type']} @ {spec['mem_bw_gbs']:,} GB/s/GPU · {link_label} · "
                                 f"platform ${plat_price:,} + ${gpu_price:,}/GPU",
            })

        DELL_SYSTEMS[key] = entry


def _validate_catalog() -> None:
    """Fail LOUDLY at import on a malformed entry.

    Without this a typo'd `gpu_spec` silently skips derivation, `calculate_tco` then raises
    KeyError on the missing `vram_gb`, and `best_fit_systems` swallows it — the system just
    vanishes from the UI with no diagnostic. At ~75 hand-entered rows that is the single most
    likely failure, so make it impossible to ship.
    """
    required = ("category", "gpus_per_node", "nvlink", "color", "system_price",
                "system_tdp_w", "gpu_model", "notes")
    for name, sys in DELL_SYSTEMS.items():
        for k in required:
            if k not in sys:
                raise ValueError(f"DELL_SYSTEMS[{name!r}] missing required key {k!r}")
        gs = sys.get("gpu_spec")
        if gs is not None and gs not in GPU_SPECS:
            raise ValueError(f"DELL_SYSTEMS[{name!r}] gpu_spec {gs!r} not in GPU_SPECS")
        if sys.get("flexible_gpus"):
            for k in ("chassis_price", "gpu_price", "chassis_tdp_w", "min_gpus", "max_gpus"):
                if k not in sys:
                    raise ValueError(f"DELL_SYSTEMS[{name!r}] flexible but missing {k!r}")
            if sys["min_gpus"] > sys["max_gpus"]:
                raise ValueError(f"DELL_SYSTEMS[{name!r}] min_gpus > max_gpus")
            # A GPU drawing 0 W is always a data error, never a real spec. This silently shipped
            # once: gpu_tdp_w is read via GPU_SPECS.get(...,0), so every SXM/HGX/NVL row reported
            # chassis-only power (an 8×B200 node at 4,800 W instead of ~12,800 W) and understated
            # OpEx across the board. Assert rather than default.
            if not sys.get("gpu_tdp_w"):
                raise ValueError(
                    f"DELL_SYSTEMS[{name!r}] gpu_tdp_w is 0/missing — GPU_SPECS[{gs!r}] needs a "
                    f"'gpu_tdp_w'. Power drives OpEx; a 0 W GPU silently understates it.")
        if sys.get("gpu_link") and sys["gpu_link"] not in _LINK_PENALTY:
            raise ValueError(f"DELL_SYSTEMS[{name!r}] unknown gpu_link {sys['gpu_link']!r}")


# The default TCO shortlist: one row per platform, each showcasing a DIFFERENT GPU, forming a
# ladder from the demo box up. Everything else is reachable from the scope selector in the UI.
DEFAULT_SYSTEMS: list[str] = [
    # --- the demo box + the mainstream ladder ---------------------------------------
    "Dell GB10",                                    # the demo box + the tok/s baseline
    "Dell PowerEdge R7715 (RTX Pro 6000 BSE)",      # flagship Blackwell PCIe
    "Dell PowerEdge R7725 (H200 NVL)",              # NVLink-bridge (2–4 way islands)
    "Dell PowerEdge R770 (L40S)",                   # Ada inference
    "Dell PowerEdge R760 (L4)",                     # 16G entry point
    # --- flexible PCIe box, both RTX Pro options ------------------------------------
    "Dell PowerEdge XE7740 (RTX Pro 6000 BSE)",
    "Dell PowerEdge XE7740 (RTX Pro 4500 BSE)",
    # --- 8-way HGX scale-up ---------------------------------------------------------
    "Dell PowerEdge XE9680 (H200 SXM5 (x8) AC)",    # H200 in SXM form (vs the NVL card above)
    "Dell PowerEdge XE9780 (B200 AC (x8))",
    "Dell PowerEdge XE9780 (B300 AC (x8))",
    # --- rack-scale (GPU-inclusive pricing) -----------------------------------------
    "Dell PowerEdge XE8712 (GB200 NVL4)",
    "Dell PowerEdge XE9712 (GB300 NVL72)",
]

# Build now, BEFORE _derive_system_specs() runs below — derivation must see the generated rows.
# _validate_catalog() is called after _LINK_PENALTY is defined (it checks gpu_link against it).
_build_catalog()

# ---------------------------------------------------------------------------
# Derive per-node specs from the GPU_SPECS reference
# ---------------------------------------------------------------------------
# Each system that names a `gpu_spec` has its aggregate VRAM, memory bandwidth,
# and TFLOPS (node-wide dense + sparse) computed as per-GPU value × gpus_per_node.
# This keeps DELL_SYSTEMS and GPU_SPECS in lockstep — and is what corrects the
# H100 NVL / 94GB memory (HBM3 @ 3.9 TB/s, not HBM2 @ 2.0). Systems without a
# `gpu_spec` (e.g. Dell GB10) keep their explicit values. The flat `tflops_fp16/8/4`
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

# Validate the catalog now that _LINK_PENALTY exists (it checks gpu_link against these keys).
_validate_catalog()
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
# Extended Model Catalog (includes models too large for Dell GB10)
# ---------------------------------------------------------------------------

MODEL_CATALOG: dict[str, dict] = {
    # Small
    "TinyLlama-1.1B":           {"params_b":    1.1, "category": "LLM",  "type": "decoder"},
    "Llama-3.2-1B":             {"params_b":    1.0, "category": "LLM",  "type": "decoder"},
    "Llama-3.2-3B":             {"params_b":    3.2, "category": "LLM",  "type": "decoder"},
    "Gemma-2-9B":               {"params_b":    9.0, "category": "LLM",  "type": "decoder"},
    # Medium
    "Mistral-7B":               {"params_b":    7.0, "category": "LLM",  "type": "decoder"},
    "Qwen2.5-7B":               {"params_b":    7.0, "category": "LLM",  "type": "decoder"},
    "Llama-3.1-8B":             {"params_b":    8.0, "category": "LLM",  "type": "decoder"},
    "Qwen3-8B":                 {"params_b":    8.0, "category": "LLM",  "type": "decoder"},
    "nvidia/Qwen3-8B-NVFP4":    {"params_b":    8.0, "category": "LLM",  "type": "decoder"},
    # Large
    "Phi-4 (14B)":              {"params_b":   14.0, "category": "LLM",  "type": "decoder"},
    "Qwen2.5-14B":              {"params_b":   14.0, "category": "LLM",  "type": "decoder"},
    "Qwen3-14B":                {"params_b":   14.0, "category": "LLM",  "type": "decoder"},
    "Mistral-Small-24B":        {"params_b":   24.0, "category": "LLM",  "type": "decoder"},
    "Gemma-2-27B":              {"params_b":   27.0, "category": "LLM",  "type": "decoder"},
    "Gemma-3-27B":              {"params_b":   27.0, "category": "LLM",  "type": "decoder"},
    "Qwen2.5-32B":              {"params_b":   32.0, "category": "LLM",  "type": "decoder"},
    "Qwen3-32B":                {"params_b":   32.0, "category": "LLM",  "type": "decoder"},
    "Mixtral-8x7B":             {"params_b":   46.7, "category": "LLM",  "type": "moe", "active_b": 12.9},
    # Extra-large (need multi-GPU or large HW)
    "Llama-3.3-70B":            {"params_b":   70.0, "category": "LLM",  "type": "decoder"},
    "Qwen2.5-72B":              {"params_b":   72.0, "category": "LLM",  "type": "decoder"},
    "Command-R+ (104B)":        {"params_b":  104.0, "category": "LLM",  "type": "decoder"},
    "Nemotron-3-Super-120B":    {"params_b":  120.0, "category": "LLM",  "type": "moe", "active_b": 12.0},
    "Mistral-Large-123B":       {"params_b":  123.0, "category": "LLM",  "type": "decoder"},
    "Mixtral-8x22B":            {"params_b":  141.0, "category": "LLM",  "type": "moe", "active_b": 39.0},
    "Qwen3-235B-A22B":          {"params_b":  235.0, "category": "LLM",  "type": "moe", "active_b": 22.0},
    # Hyperscale (multi-node only)
    "Nemotron-4-340B":          {"params_b":  340.0, "category": "LLM",  "type": "decoder"},
    "Llama-3.1-405B":           {"params_b":  405.0, "category": "LLM",  "type": "decoder"},
    "Llama-3.1-405B (FP8)":     {"params_b":  405.0, "category": "LLM",  "type": "decoder"},
    "DeepSeek-V3 (671B)":       {"params_b":  671.0, "category": "LLM",  "type": "moe",
                                 "active_b": 37.0, "precisions": ["FP8", "FP4"], "native": "FP8"},
    "DeepSeek-R1 (671B)":       {"params_b":  671.0, "category": "LLM",  "type": "moe",
                                 "active_b": 37.0, "precisions": ["FP8", "FP4"], "native": "FP8"},
    # VLM / CNN
    "CLIP ViT-L/14":            {"params_b":    0.4, "category": "VLM",  "type": "encoder"},
    "ResNet-50":                {"params_b":    0.03,"category": "CNN",  "type": "encoder"},
    "EfficientNet-B4":          {"params_b":    0.02,"category": "CNN",  "type": "encoder"},
}

# Precisions a model can actually be served at. Explicit per-entry `precisions` win;
# otherwise inferred from the shipped checkpoint form (pre-quantized NVFP4 / FP8-native)
# and category. This is the single source of truth for the precision selectors.
_STD_LLM_PREC = ["BF16", "FP16", "FP8", "INT8", "FP4", "FP32"]
_VISION_PREC  = ["FP16", "BF16", "INT8"]


def supported_precisions(model_name: str) -> list[str]:
    """Precisions this catalog model actually runs at (drives the TCO precision picker)."""
    info = MODEL_CATALOG.get(model_name, {})
    if info.get("precisions"):
        return list(info["precisions"])
    name = (model_name or "").upper()
    if "NVFP4" in name or "NVF4" in name:
        return ["NVFP4", "FP4"]          # pre-quantized 4-bit — only loads at FP4/NVFP4
    if "FP8" in name:
        return ["FP8", "FP4"]            # FP8-native checkpoint
    if info.get("category") in ("VLM", "CNN"):
        return list(_VISION_PREC)
    return list(_STD_LLM_PREC)


def native_precision(model_name: str) -> str:
    """The precision the model ships / is designed for (the default selection)."""
    info = MODEL_CATALOG.get(model_name, {})
    if info.get("native"):
        return info["native"]
    # else the shipped/designed form = the first supported precision
    # (_STD_LLM_PREC→BF16, _VISION_PREC→FP16, NVFP4 list→NVFP4).
    sup = supported_precisions(model_name)
    return sup[0] if sup else "FP16"

_BYTES_PER_PARAM = {
    "FP64": 8.0, "FP32": 4.0, "TF32": 4.0, "FP16": 2.0, "BF16": 2.0,  # TF32 stores as FP32 (4B)
    "INT8": 1.0, "FP8": 1.0, "FP4": 0.5, "NVFP4": 0.5,
}

GB10_BW_GBS = 273  # Dell GB10 real LPDDR5X unified-memory bandwidth (NOT the 900 GB/s
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
# low-bandwidth part (e.g. Dell GB10's 128 GB @ 273 GB/s) gets "free" aggregate just by
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
# boxes gets no cross-node batching. So a 200-desktop Dell GB10 fleet shouldn't cost-model
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
# Workforce demand model — total employees → effective concurrent sessions
# ---------------------------------------------------------------------------
# TCO sizing is no longer a flat "concurrent users" count. An enterprise seat
# is split into three usage tiers, and load is driven by PER-TIER concurrency
# (how often each tier has a live session), not headcount alone — so a
# power-heavy workforce sizes above a flat baseline. A handful of power users
# also dominate token consumption; the INTENSITY weights (relative tokens per
# active session) express that for the token-share view.
#
# effective_sessions = Σ round(employees · headcount%_tier · concurrency%_tier)
# token_share_tier   ∝ active_sessions_tier · INTENSITY_tier   (normalized)
#
# Defaults: Power ~8% of seats but ~70% active and 4× the tokens/session →
# ~55–60% of all tokens from a thin sliver of the workforce. General is the
# 20% baseline; Minimal is rarely active and light.

WORKFORCE_TIER_ORDER = ("power", "general", "minimal")

WORKFORCE_DEFAULTS = {
    # headcount % of employees (general is derived = 100 − power − minimal)
    "power_pct":     8.0,
    "minimal_pct":  30.0,
    # per-tier concurrency: fraction of that tier with a live session at peak
    "conc_power":   70.0,
    "conc_general": 20.0,   # the baseline concurrency
    "conc_minimal":  5.0,
}

# Relative tokens per active session (general = 1.0). Shapes the token-share
# bar only; sizing is session-count based (per-tier concurrency).
WORKFORCE_INTENSITY = {"power": 4.0, "general": 1.0, "minimal": 0.4}


@dataclass
class WorkforceDemand:
    total_employees:       int
    headcount:             dict          # tier -> employees (int)
    headcount_pct:         dict          # tier -> % of workforce
    active_sessions:       dict          # tier -> concurrent sessions (int)
    token_share:           dict          # tier -> % of tokens (sums ~100)
    effective_sessions:    int           # Σ active_sessions, ≥ 1 (feeds num_users)
    effective_concurrency: float         # effective_sessions / employees (fraction)
    general_pct:           float         # derived headcount % for the general tier
    warning:               str = ""      # non-empty if the mix was clamped


def workforce_demand(
    total_employees: int,
    power_pct:     float = WORKFORCE_DEFAULTS["power_pct"],
    minimal_pct:   float = WORKFORCE_DEFAULTS["minimal_pct"],
    conc_power:    float = WORKFORCE_DEFAULTS["conc_power"],
    conc_general:  float = WORKFORCE_DEFAULTS["conc_general"],
    conc_minimal:  float = WORKFORCE_DEFAULTS["conc_minimal"],
    intensity:     Optional[dict] = None,
) -> WorkforceDemand:
    """Convert a total-employee count + usage-tier mix into an effective
    concurrent-session count for TCO sizing, plus a per-tier token-share split.

    All *_pct args are percentages (0–100). General headcount is the remainder
    (100 − power − minimal); if power + minimal exceed 100 it is clamped to 0
    and a warning is returned rather than raising.
    """
    N   = max(int(total_employees), 1)
    wt  = intensity or WORKFORCE_INTENSITY

    warning = ""
    gen_pct = 100.0 - power_pct - minimal_pct
    if gen_pct < 0:
        warning = (f"Power ({power_pct:.0f}%) + Minimal ({minimal_pct:.0f}%) exceed 100% — "
                   f"General clamped to 0%.")
        gen_pct = 0.0

    pct = {"power": power_pct, "general": gen_pct, "minimal": minimal_pct}
    conc = {"power": conc_power, "general": conc_general, "minimal": conc_minimal}

    headcount       = {t: int(round(N * pct[t] / 100.0)) for t in WORKFORCE_TIER_ORDER}
    active_sessions = {t: int(round(N * pct[t] / 100.0 * conc[t] / 100.0))
                       for t in WORKFORCE_TIER_ORDER}

    eff = max(1, sum(active_sessions.values()))

    weighted = {t: active_sessions[t] * wt.get(t, 1.0) for t in WORKFORCE_TIER_ORDER}
    wtot     = sum(weighted.values())
    if wtot > 0:
        token_share = {t: weighted[t] / wtot * 100.0 for t in WORKFORCE_TIER_ORDER}
    else:
        token_share = {t: 0.0 for t in WORKFORCE_TIER_ORDER}

    return WorkforceDemand(
        total_employees=N,
        headcount=headcount,
        headcount_pct=pct,
        active_sessions=active_sessions,
        token_share=token_share,
        effective_sessions=eff,
        effective_concurrency=eff / N,
        general_pct=gen_pct,
        warning=warning,
    )


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
    Scale measured Dell GB10 throughput to a target system.
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
