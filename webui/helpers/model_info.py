"""
Curated reference for every model and HPC test used in the benchmark scenarios.

Always-available (offline) descriptions: architecture, parameter count, precision
behavior, and what each one stresses on the Dell GB10. Keyed by the exact dropdown
strings used in streamlit_app.py; lookup also tolerates the bare name after "/".
"""

# Each entry: {"headline": one-liner, "body": markdown bullets}
MODEL_INFO: dict[str, dict] = {
    # ----------------------------------------------------------------- LLMs
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "headline": "1.1B Llama-architecture chat model — the latency floor",
        "body": (
            "- **Architecture:** decoder-only transformer (Llama 2 layout), 22 layers, "
            "grouped-query attention, 2048 hidden.\n"
            "- **Params:** 1.1B · trained on ~3T tokens · chat fine-tuned.\n"
            "- **Footprint:** ~2.2 GB FP16 — fits every precision with room to spare.\n"
            "- **On Dell GB10:** memory-trivial; useful for measuring the **framework / TTFT floor** "
            "and warming the GPU. Decode is so light it barely touches the LPDDR5X bandwidth ceiling."
        ),
    },
    "unsloth/Llama-3.2-3B-Instruct": {
        "headline": "Llama 3.2 3B Instruct — the on-device narrator; small but capable",
        "body": (
            "- **Architecture:** decoder-only transformer (Llama 3.2 layout), grouped-query "
            "attention, RoPE, 128K context.\n"
            "- **Params:** 3.2B · instruction-tuned · distilled from the larger Llama 3 models.\n"
            "- **Footprint:** ~6.5 GB BF16 / ~3.5 GB INT8 / ~2 GB FP4 — trivially light on the "
            "128 GB unified pool.\n"
            "- **On Dell GB10:** the **base on-device narrator** (runs the talking-point generation at "
            "BF16). Decode is memory-bandwidth bound; far better narration quality than the 1.1B "
            "TinyLlama it replaces while still loading in seconds and leaving the pool wide open.\n"
            "- *Weights via the ungated unsloth re-host of meta-llama/Llama-3.2-3B-Instruct.*"
        ),
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "headline": "Alibaba Qwen2.5 7B — mid-size multilingual workhorse",
        "body": (
            "- **Architecture:** dense decoder-only transformer, grouped-query attention, "
            "RoPE, up to 128K context (YaRN).\n"
            "- **Params:** 7.6B · strong multilingual, math, and coding.\n"
            "- **Footprint:** ~15 GB FP16 / ~8 GB INT8 / ~4 GB FP4.\n"
            "- **On Dell GB10:** comfortable single-user latency target; decode is memory-bandwidth bound, "
            "so FP16→FP4 mainly buys VRAM headroom and a modest speedup on the 273 GB/s pool."
        ),
    },
    "mistralai/Mistral-7B-v0.1": {
        "headline": "Mistral 7B (base) — efficient sliding-window attention",
        "body": (
            "- **Architecture:** decoder-only with **sliding-window attention** (4096) + GQA — "
            "cheaper long-context attention than full attention.\n"
            "- **Params:** 7.2B · base (non-instruct) checkpoint.\n"
            "- **Footprint:** ~14 GB FP16.\n"
            "- **On Dell GB10:** a clean 7B reference; pairs well with the 7B Qwen for cross-architecture "
            "throughput comparison at the same parameter scale."
        ),
    },
    "microsoft/Phi-4": {
        "headline": "Microsoft Phi-4 (14B) — reasoning-dense, synthetic-data trained",
        "body": (
            "- **Architecture:** dense decoder-only transformer.\n"
            "- **Params:** 14B · trained heavily on curated + synthetic data for outsized "
            "reasoning per parameter.\n"
            "- **Footprint:** ~28 GB FP16 / ~14 GB INT8.\n"
            "- **On Dell GB10:** fits FP16 with ample headroom; good 'smart mid-size' demo where quality "
            "per GB matters more than raw size."
        ),
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "headline": "Qwen2.5 14B — dense, long-context instruct model",
        "body": (
            "- **Architecture:** dense decoder-only, grouped-query attention, long context.\n"
            "- **Params:** 14.7B.\n"
            "- **Footprint:** ~28 GB FP16 / ~15 GB INT8.\n"
            "- **On Dell GB10:** fits FP16; a step up in quality from the 7B class at ~2× the memory and "
            "decode cost."
        ),
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "headline": "Mixtral 8×7B — sparse Mixture-of-Experts (the memory-vs-compute lesson)",
        "body": (
            "- **Architecture:** **sparse MoE** — 8 expert FFNs per layer, top-2 routing. "
            "All experts must be resident in memory, but only 2 run per token.\n"
            "- **Params:** ~46.7B total weights · **~12.9B active** per token.\n"
            "- **Footprint:** ~93 GB FP16 (all experts loaded) — the headline demo of why **VRAM "
            "capacity ≠ compute**. Compute behaves like a ~13B; memory behaves like a 47B.\n"
            "- **On Dell GB10:** FP16 is blocked by the memory guard (>100 GB usable); runs at **INT8 "
            "(~51 GB)** or **FP4 (~26 GB)** — a great 'only fits when quantized' talking point."
        ),
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "headline": "Qwen2.5 32B — largest dense model that fits Dell GB10 at FP16",
        "body": (
            "- **Architecture:** dense decoder-only, grouped-query attention.\n"
            "- **Params:** 32.5B.\n"
            "- **Footprint:** ~64 GB FP16 / ~33 GB INT8 — fits the Dell GB10's 128 GB unified pool at FP16.\n"
            "- **On Dell GB10:** the upper bound for single-device FP16 inference here; decode is firmly "
            "memory-bandwidth bound, so it best showcases the value of the large unified pool."
        ),
    },
    "nvidia/Qwen3-8B-NVFP4": {
        "headline": "NVIDIA Qwen3-8B NVFP4 — native 4-bit Blackwell inference",
        "body": (
            "- **Architecture:** Qwen3 8B decoder, **pre-quantized to NVFP4** with NVIDIA ModelOpt "
            "(weights packed 4-bit, BF16 compute).\n"
            "- **Params:** 8B · ships already quantized — load as-is, no runtime bitsandbytes pass.\n"
            "- **Footprint:** ~5 GB.\n"
            "- **On Dell GB10:** the **showcase for Blackwell hardware-accelerated FP4** — only valid at "
            "FP4/NVFP4 precision (loading it as FP16 produces garbage). Demonstrates the 4-bit "
            "memory + throughput win the Dell GB10's tensor cores are built for."
        ),
    },

    # ---------------------------------------------------------------- Vision / CNN
    "openai/clip-vit-base-patch32": {
        "headline": "CLIP ViT-B/32 — contrastive image+text encoder",
        "body": (
            "- **Architecture:** Vision Transformer image encoder (32×32 patches) + text encoder, "
            "trained contrastively to a shared embedding space.\n"
            "- **Params:** ~151M.\n"
            "- **On Dell GB10:** tiny; the benchmark drives the image tower with synthetic 224×224 batches "
            "to measure encode throughput (images/sec) and how it scales with batch size."
        ),
    },
    "openai/clip-vit-large-patch14": {
        "headline": "CLIP ViT-L/14 — higher-accuracy CLIP, finer patches",
        "body": (
            "- **Architecture:** larger ViT (14×14 patches → more tokens per image) + text encoder.\n"
            "- **Params:** ~428M.\n"
            "- **On Dell GB10:** ~3× the encoder compute of ViT-B/32; shows how finer patch size raises "
            "both accuracy and per-image cost at the same resolution."
        ),
    },
    "google/vit-base-patch16-224": {
        "headline": "ViT-B/16 — supervised image classification transformer",
        "body": (
            "- **Architecture:** Vision Transformer, 16×16 patches @224, supervised ImageNet head.\n"
            "- **Params:** ~86M.\n"
            "- **On Dell GB10:** classic ViT throughput reference; 16px patches → 196 tokens/image, "
            "heavier than ViT-B/32's 49."
        ),
    },
    "facebook/dino-vits16": {
        "headline": "DINO ViT-S/16 — self-supervised features (no labels)",
        "body": (
            "- **Architecture:** small ViT (16×16 patches) trained with **DINO self-distillation** — "
            "learns strong visual features without labels.\n"
            "- **Params:** ~21M.\n"
            "- **On Dell GB10:** the lightest vision model here; useful as the speed/cost floor for the "
            "image pipeline."
        ),
    },
    "microsoft/resnet-50": {
        "headline": "ResNet-50 — the canonical convolutional baseline",
        "body": (
            "- **Architecture:** 50-layer **CNN** with residual blocks (not a transformer).\n"
            "- **Params:** ~25M.\n"
            "- **On Dell GB10:** provides the **CNN-vs-Transformer** contrast — convolutions are compute- "
            "and cache-friendly in a different way than attention, so its throughput/precision curve "
            "differs from the ViT/CLIP models."
        ),
    },
    "google/efficientnet-b4": {
        "headline": "EfficientNet-B4 — compound-scaled efficient CNN",
        "body": (
            "- **Architecture:** CNN scaled jointly in depth/width/resolution (compound scaling); "
            "mobile inverted-bottleneck blocks.\n"
            "- **Params:** ~19M.\n"
            "- **On Dell GB10:** best accuracy-per-FLOP CNN in the set; pairs with ResNet-50 to show how "
            "architecture efficiency — not just size — moves images/sec."
        ),
    },

    # ---------------------------------------------------------------- HPC / Quant tests
    "MatMul Benchmark": {
        "headline": "Dense GEMM — peak compute (TFLOPS)",
        "body": (
            "- **What it is:** large square matrix multiply (n×n · n×n).\n"
            "- **Bound by:** **GPU compute** — O(n³) math vs O(n²) memory, so it saturates the tensor "
            "cores rather than the memory bus.\n"
            "- **Reports:** sustained TFLOPS. Scales strongly with precision — BF16/FP16 tensor cores "
            "vastly outrun FP32, and FP4 outruns those again on Blackwell.\n"
            "- **On Dell GB10:** the headline number for the chip's math throughput."
        ),
    },
    "Bandwidth Test": {
        "headline": "Streaming elementwise — memory bandwidth (GB/s)",
        "body": (
            "- **What it is:** read-modify-write passes over a multi-GB tensor (×2, +1, sqrt).\n"
            "- **Bound by:** **memory bandwidth** — almost no math per byte.\n"
            "- **Reports:** effective GB/s against the Dell GB10's ~273 GB/s LPDDR5X ceiling.\n"
            "- **On Dell GB10:** directly measures the unified-memory bandwidth that gates LLM decode."
        ),
    },
    "LOB Bandwidth": {
        "headline": "Limit Order Book — irregular scatter/gather (FinTech)",
        "body": (
            "- **What it is:** simulates market-data access — gather random price levels → compute "
            "best-bid/offer spread → scatter-add updates back.\n"
            "- **Bound by:** **memory bandwidth + latency** under a random (non-streaming) access pattern.\n"
            "- **Reports:** effective GB/s, plus a gather/compute/scatter phase breakdown.\n"
            "- **On Dell GB10:** models real exchange/HFT data movement, where access is irregular rather "
            "than nicely sequential."
        ),
    },
    "Reduction Ops": {
        "headline": "Vector reductions — bandwidth-bound aggregation",
        "body": (
            "- **What it is:** sum / max / mean / std over a very large vector.\n"
            "- **Bound by:** **memory bandwidth** — each element is read once, output is a scalar.\n"
            "- **On Dell GB10:** complements the Bandwidth Test with a reduction access pattern."
        ),
    },
    "Fill Memory (60%)": {
        "headline": "High-occupancy fill — capacity + sustained bandwidth",
        "body": (
            "- **What it is:** allocates ~60% of VRAM then runs sustained ops on it.\n"
            "- **Bound by:** **memory capacity and bandwidth at high occupancy** (where caching helps least).\n"
            "- **On Dell GB10:** shows the chip holds bandwidth even when the 128 GB pool is heavily filled."
        ),
    },
    "Dual Model Serving": {
        "headline": "Two concurrent model tensors — multi-tenant memory",
        "body": (
            "- **What it is:** loads two model-sized tensors and runs them on separate **CUDA streams**.\n"
            "- **Bound by:** **memory capacity + concurrency** — can the unified pool host multiple "
            "models at once, and do they overlap?\n"
            "- **On Dell GB10:** the multi-tenant / co-location story for the large unified memory."
        ),
    },
    "Black-Scholes Options": {
        "headline": "Vectorized option pricing — compute-heavy quant finance",
        "body": (
            "- **What it is:** Black-Scholes call/put pricing over millions of contracts using "
            "log / exp / erf / sqrt.\n"
            "- **Bound by:** **GPU compute** — transcendental math dominates (memory is a smaller factor).\n"
            "- **Reports:** millions of options/sec and GFLOPS.\n"
            "- **On Dell GB10:** a real derivatives-pricing workload, not a synthetic stub."
        ),
    },
    "Monte Carlo VaR": {
        "headline": "Monte Carlo Value-at-Risk — risk simulation",
        "body": (
            "- **What it is:** simulate ~1M geometric-Brownian-motion portfolio paths over a 10-day "
            "horizon, then take the 99th-percentile loss.\n"
            "- **Bound by:** **memory bandwidth** — generating and reducing a huge random shock tensor "
            "dominates.\n"
            "- **Reports:** millions of paths/sec and the VaR estimate.\n"
            "- **On Dell GB10:** a representative bank/treasury risk workload that leans on the large pool."
        ),
    },
}


# Precision a model is released in / designed for, with a one-line rationale.
# (label, note) — used in the on-device talking points and the Settings card.
DESIGNED_PRECISION: dict[str, tuple[str, str]] = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": ("BF16", "trained in BF16; tiny enough to run any precision well"),
    "unsloth/Llama-3.2-3B-Instruct":      ("BF16", "released in BF16; the on-device narrator runs it at BF16 for quality"),
    "Qwen/Qwen2.5-7B-Instruct":           ("BF16", "released in BF16; INT8/FP4 trade a little quality for memory"),
    "mistralai/Mistral-7B-v0.1":          ("BF16", "released in BF16"),
    "microsoft/Phi-4":                    ("BF16", "released in BF16"),
    "Qwen/Qwen2.5-14B-Instruct":          ("BF16", "released in BF16"),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ("BF16", "BF16-native MoE — on Dell GB10 it only fits at INT8 or FP4"),
    "Qwen/Qwen2.5-32B-Instruct":          ("BF16", "BF16-native; fits FP16 within Dell GB10's 128 GB pool"),
    "nvidia/Qwen3-8B-NVFP4":              ("NVFP4", "pre-quantized 4-bit for Blackwell FP4 — only valid at FP4/NVFP4"),
    # Vision / CNN (informational; these don't self-narrate)
    "openai/clip-vit-base-patch32":  ("FP16", "FP16/BF16 is the sweet spot; INT8 for max image throughput"),
    "openai/clip-vit-large-patch14": ("FP16", "FP16/BF16 sweet spot"),
    "google/vit-base-patch16-224":   ("FP16", "FP16/BF16 sweet spot"),
    "facebook/dino-vits16":          ("FP16", "FP16/BF16 sweet spot"),
    "microsoft/resnet-50":           ("FP16", "FP16/BF16; INT8 common for CNN deployment"),
    "google/efficientnet-b4":        ("FP16", "FP16/BF16; INT8 common for edge CNN deployment"),
}


def designed_precision(name: str) -> tuple[str | None, str]:
    """Return (precision_label, rationale) the model is designed for, or (None, '')."""
    if not name:
        return None, ""
    if name in DESIGNED_PRECISION:
        return DESIGNED_PRECISION[name]
    bare = name.split("/")[-1].lower()
    for key, val in DESIGNED_PRECISION.items():
        if key.split("/")[-1].lower() == bare:
            return val
    return None, ""


# Precisions delivered by on-the-fly quantization (bitsandbytes / packed 4-bit).
# These stay available on standard checkpoints — they are NOT an incompatibility.
QUANT_PRECISIONS = frozenset({"INT8", "FP4", "NVFP4"})


def precision_compatible(model_name: str, precision: str) -> tuple[bool, str]:
    """Whether a precision can actually run on this model's checkpoint.

    Returns (ok, reason). Quantization (INT8/FP4/NVFP4) on a standard model stays
    available. The hard incompatibility today is a **pre-quantized NVFP4
    checkpoint**, whose packed 4-bit weights only load at FP4/NVFP4 — loading it
    as FP16/FP32/BF16/INT8 produces shape mismatches / garbage, so we mark it
    Not Compatible rather than silently switching precision and running.
    """
    name = (model_name or "").upper()
    prequant_nvfp4 = "NVFP4" in name or "NVF4" in name
    if prequant_nvfp4 and precision.upper() not in ("FP4", "NVFP4"):
        return False, "pre-quantized NVFP4 checkpoint — only runs at FP4/NVFP4"
    return True, ""


def lookup_model_info(name: str) -> dict | None:
    """Find the curated entry for a model/test name, tolerating org-prefix forms."""
    if not name:
        return None
    if name in MODEL_INFO:
        return MODEL_INFO[name]
    # try bare name after "/"
    bare = name.split("/")[-1]
    for key, val in MODEL_INFO.items():
        if key.split("/")[-1].lower() == bare.lower():
            return val
    # case-insensitive exact
    low = name.lower()
    for key, val in MODEL_INFO.items():
        if key.lower() == low:
            return val
    return None
