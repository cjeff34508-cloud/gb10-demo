"""
Dell Pro Max GB10 Demo Suite — Failure-Mode Benchmark
Organised by scenario (not modality): each run surfaces workload phase,
primary bottleneck, operating condition, and business-relevant output.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch
from pathlib import Path

from helpers import (
    LLMInference,
    VisionModelInference,
    HPCBenchmark,
    BenchmarkMetrics,
    MemoryStress,
    bottleneck_badge_html,
    BOTTLENECK_COLORS,
    get_gpu_stats,
)
from helpers.mem_guard import check_precisions, estimate_model_gb, GB10_USABLE_GB, GB10_RESERVE_GB
from helpers.benchmark_utils import free_cuda_memory
from helpers.tco_engine import (
    DELL_SYSTEMS, DEFAULT_SYSTEMS, MODEL_CATALOG, calculate_tco, best_fit_systems, format_usd,
    model_memory_gb, GB10_BW_GBS, assign_ratings, RATING_COLORS, RATING_ORDER,
    MC_PATH_PRESETS, MC_BW_EFF, calculate_tco_montecarlo,
    supported_precisions, native_precision,
    workforce_demand, WORKFORCE_DEFAULTS,
)
from helpers.model_info import lookup_model_info, designed_precision, precision_compatible
from helpers.on_device_ai import narrator_available, make_narrator, NARRATOR_MODEL, NARRATOR_PRECISION
from helpers.cloud_pricing import api_token_costs

# Display order for precisions: lowest bit-width first → highest.
_PRECISION_ORDER = ["FP4", "NVFP4", "INT4", "FP8", "INT8", "FP16", "BF16", "TF32", "FP32", "FP64"]


def _precision_rank(precision: str) -> int:
    """Rank a precision label lowest→highest bit-width. Tolerates suffixed labels
    like 'FP32 (no INT8 support)'. Unknown precisions sort to the end."""
    key = (precision or "").split()[0].upper() if precision else ""
    try:
        return _PRECISION_ORDER.index(key)
    except ValueError:
        return len(_PRECISION_ORDER)


def _local_tp_prompt(model, precision, tps, ttft, mem_pct, bottleneck,
                     designed=None, designed_note=""):
    """Prompt for the on-device model to narrate its own benchmark run."""
    prec_line = ""
    if designed:
        prec_line = (
            f" You are designed for / work best at {designed} precision"
            + (f" ({designed_note})" if designed_note else "")
            + ". One of your three bullets MUST state the precision you are designed for "
            "and why that matters on this hardware."
        )
    return (
        f"You are {model.split('/')[-1]}, an AI model summarizing your OWN benchmark "
        f"run. You are running locally on an Dell Pro Max GB10 (Grace-Blackwell, 128 GB unified "
        f"memory) at {precision} precision. Measured results: {tps:.0f} tokens/sec decode, "
        f"{ttft:.0f} ms time-to-first-token, {mem_pct:.0f}% of unified memory used, primary "
        f"bottleneck '{bottleneck}'.{prec_line} Write exactly three short bullet-point "
        f"talking points about running AI locally on this hardware, then one bold one-line "
        f"takeaway. Be concise and concrete, and do not invent numbers beyond those given."
    )

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dell Pro Max GB10 Demo Suite — Dell Technologies",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Dell Technologies Brand Theme ─────────────────────────────────── */

/* Sidebar — Dell Dark Navy */
[data-testid="stSidebar"] { background-color: #003576 !important; }

/* Sidebar — general text (labels, captions, markdown) */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stMultiSelect > label,
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stNumberInput > label,
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color: #FFFFFF !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stMultiSelect > label { color: #C8D8E8 !important; font-size: 0.85em; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }

/* Sidebar — selectbox & multiselect input boxes: dark bg, white text, white arrow */
[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
    background-color: rgba(255,255,255,0.12) !important;
    border-color: rgba(255,255,255,0.35) !important;
    border-radius: 4px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="value"],
[data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="value"] *,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] [aria-selected] {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="icon"],
[data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="icon"] svg,
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #FFFFFF !important; color: #FFFFFF !important; }

/* Sidebar — multiselect tags */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: #007DB8 !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span { color: #FFFFFF !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] button svg { fill: #FFFFFF !important; }

/* Sidebar — number input (batch size, start/max batch): dark field + white text.
   Target the input element itself (not just the wrapper) so it can't render white-on-white. */
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stSidebar"] [data-testid="stNumberInput"] div[data-baseweb="base-input"] {
    background-color: #14315F !important;
    border-color: rgba(255,255,255,0.35) !important;
}
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    color: #FFFFFF !important;
    background-color: #14315F !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
/* Number-input +/- stepper buttons */
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background-color: rgba(255,255,255,0.18) !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg { fill: #FFFFFF !important; }

/* Sidebar — slider track and thumb */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background-color: #007DB8 !important; border-color: #007DB8 !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMax"] { color: #C8D8E8 !important; }

/* Dropdown popover list (renders outside sidebar, needs global scope) */
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul[role="listbox"] {
    background-color: #002A5C !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
}
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] li[role="option"] {
    background-color: #002A5C !important;
    color: #FFFFFF !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] li[role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background-color: #007DB8 !important;
    color: #FFFFFF !important;
}
[data-baseweb="popover"] [role="option"] * { color: #FFFFFF !important; }

/* ── Main-area dropdowns (selectbox / multiselect) — high-contrast dark text ──
   Sidebar selects keep their white-on-navy styling via the more-specific
   [data-testid="stSidebar"] rules above; these apply only to the content area. */
.stApp [data-baseweb="select"] > div:first-child {
    background-color: #FFFFFF !important;
    border-color: #8FA8C0 !important;
}
.stApp [data-baseweb="select"] [data-baseweb="value"],
.stApp [data-baseweb="select"] [data-baseweb="value"] *,
.stApp [data-baseweb="select"] input,
.stApp [data-baseweb="select"] [aria-selected] { color: #1D1D1B !important; }
.stApp [data-baseweb="select"] [data-baseweb="icon"] svg,
.stApp [data-baseweb="select"] svg { fill: #003576 !important; color: #003576 !important; }
/* Selectbox / multiselect field labels — dark navy, clearly legible */
.stApp [data-testid="stWidgetLabel"] p,
.stApp .stSelectbox > label,
.stApp .stMultiSelect > label,
.stApp .stNumberInput > label,
.stApp .stSlider > label { color: #1D1D1B !important; font-weight: 600; }
/* Main-area multiselect tags — Dell blue chip, white text */
.stApp [data-baseweb="tag"] { background-color: #007DB8 !important; }
.stApp [data-baseweb="tag"] span,
.stApp [data-baseweb="tag"] span * { color: #FFFFFF !important; }
.stApp [data-baseweb="tag"] button svg { fill: #FFFFFF !important; }
/* Sidebar labels must stay light — re-assert after the .stApp label rule above */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stMultiSelect > label,
[data-testid="stSidebar"] .stNumberInput > label,
[data-testid="stSidebar"] .stSlider > label { color: #C8D8E8 !important; }

/* Main background */
.stApp { background-color: #F4F5F7; }
.main .block-container { background-color: #F4F5F7; }

/* Dell-style header banner */
.dell-header {
    background: linear-gradient(135deg, #003576 0%, #007DB8 100%);
    color: white;
    padding: 20px 28px 16px 28px;
    border-radius: 6px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 8px rgba(0,53,118,0.25);
}
.dell-header .dell-logo {
    font-size: 1.1em;
    font-weight: 300;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.9;
}
.dell-header .dell-logo strong { font-weight: 800; letter-spacing: 0.05em; }
.dell-header .dell-title {
    font-size: 1.55em;
    font-weight: 700;
    letter-spacing: 0.01em;
}
.dell-header .dell-subtitle {
    font-size: 0.82em;
    opacity: 0.75;
    margin-top: 2px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.dell-header .dell-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 0.78em;
    text-align: center;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Metrics — Dell Blue left border */
[data-testid="metric-container"] {
    background: white;
    border-left: 4px solid #007DB8;
    border-radius: 4px;
    padding: 12px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] { color: #003576 !important; font-weight: 600; font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #1D1D1B !important; }

/* Primary button — Dell Blue */
.stButton > button[kind="primary"] {
    background-color: #007DB8 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
}
.stButton > button[kind="primary"]:hover { background-color: #005B8E !important; }

/* Tabs — Dell Blue active indicator */
button[data-baseweb="tab"][aria-selected="true"] {
    color: #007DB8 !important;
    border-bottom-color: #007DB8 !important;
}

/* Section headers */
h1 { color: #003576 !important; }
h2, h3 { color: #007DB8 !important; }

/* Download button */
.stDownloadButton > button {
    background-color: #007DB8 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover { background-color: #005B8E !important; }

/* Biz card */
.biz-card {
    background: white;
    border: 1px solid #D0DCE8;
    border-left: 4px solid #007DB8;
    padding: 18px 24px;
    border-radius: 4px;
    margin-bottom: 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.biz-card h4 { color: #003576; margin: 0 0 12px 0; font-size: 0.9em; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700; }
.cond-table td { padding: 3px 12px; font-size: 0.88em; }
.cond-table td:first-child { color: #666; }

/* Dell footer */
.dell-footer {
    text-align: center;
    color: #888;
    font-size: 11px;
    padding: 8px 0;
    border-top: 1px solid #D0DCE8;
    letter-spacing: 0.05em;
}
.dell-footer span { color: #007DB8; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = {}

# ---------------------------------------------------------------------------
# Sidebar — GPU info
# ---------------------------------------------------------------------------
st.sidebar.markdown("""
<div style='padding:0 0 12px 0;border-bottom:1px solid rgba(255,255,255,0.2);margin-bottom:12px;'>
  <div style='font-size:1.05em;font-weight:800;letter-spacing:0.06em;color:white;'>DELL</div>
  <div style='font-size:0.65em;font-weight:300;letter-spacing:0.18em;color:#C8D8E8;text-transform:uppercase;'>Technologies</div>
  <div style='font-size:0.72em;color:#A0BCD4;margin-top:6px;letter-spacing:0.04em;'>Dell Pro Max GB10 Demo Suite</div>
</div>
<div style='font-size:0.75em;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#C8D8E8;margin-bottom:6px;'>⚙ Configuration</div>
""", unsafe_allow_html=True)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"✓ {gpu_name}")
    st.sidebar.info(f"CUDA {torch.version.cuda}")
    mem_alloc = torch.cuda.memory_allocated() / 1024 / 1024
    mem_res = torch.cuda.memory_reserved() / 1024 / 1024
    st.sidebar.metric("GPU Memory", f"{mem_alloc:.0f} / {mem_res:.0f} MB")
else:
    st.sidebar.error("⚠ No GPU detected")

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Scenario selector
# ---------------------------------------------------------------------------
SCENARIOS = {
    "Quick Inference":       "Single user, short context, batch 1 — baseline latency & TTFT",
    "Long Context":          "Single user, long context (4K tokens) — prefill scaling",
    "Batch Throughput":      "Single user, large batch — peak tokens/sec, up to OOM",
    "Batch Limit Sweep":     "Auto-escalate batch until OOM — finds Dell Pro Max GB10 throughput ceiling",
    "Multi-User Concurrency":"N simultaneous sessions — max practical sessions",
    "Memory Pressure":       "Load until VRAM fills — fits-in-memory vs spill",
    "Vision Throughput":     "Images/sec across precisions — CLIP / ViT",
    "HPC / Quant Analysis":  "MatMul, bandwidth, LOB — TFLOPS & GB/s",
}

scenario = st.sidebar.selectbox("Scenario", list(SCENARIOS.keys()))
st.sidebar.caption(SCENARIOS[scenario])
st.sidebar.divider()

# ---------------------------------------------------------------------------
# Per-scenario sidebar controls
# ---------------------------------------------------------------------------
models_dir = Path.home() / "gb10-demo" / "models"
_incompat: dict[str, str] = {}   # precision -> reason; Not-Compatible, never run

if scenario in ("Quick Inference", "Long Context", "Batch Throughput", "Batch Limit Sweep"):
    # Only models present on disk that fit within Dell Pro Max GB10 usable memory (100 GB) at
    # one or more precisions. 70B/72B are excluded: they exceed FP16/BF16 capacity
    # and are not downloaded. (See TCO Analysis tab to plan larger models on bigger HW.)
    llm_models = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "unsloth/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-v0.1",
        "microsoft/Phi-4",
        "Qwen/Qwen2.5-14B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen2.5-32B-Instruct",
        "nvidia/Qwen3-8B-NVFP4",
    ]
    selected_model = st.sidebar.selectbox("Model", llm_models)
    # Limit precisions to what this checkpoint actually runs at: a pre-quantized NVFP4
    # model loads only at FP4/NVFP4; standard BF16 checkpoints run FP32/FP16/BF16 + FP8/INT8/FP4.
    # FP8 = real e4m3 tensor-core compute via torchao (validated on this Dell Pro Max GB10, sm_121 aarch64);
    # INT8/FP4 via bitsandbytes. NVFP4 is not offered for standard models (needs a packed checkpoint).
    _nvfp4_model = "NVFP4" in selected_model.upper()
    precision_opts = ["FP4", "NVFP4"] if _nvfp4_model else ["FP32", "FP16", "BF16", "FP8", "INT8", "FP4"]
    _default_prec = ["NVFP4"] if _nvfp4_model else ["FP16", "BF16"]
    precisions = st.sidebar.multiselect("Precisions", precision_opts, default=_default_prec)

    # Precision compatibility — a precision the checkpoint can't actually run
    # (e.g. pre-quantized NVFP4 at FP16) is marked Not Compatible and excluded;
    # we do NOT switch it to another precision and run. Quantization stays.
    _compatible = []
    for _p in precisions:
        _ok, _why = precision_compatible(selected_model, _p)
        if _ok:
            _compatible.append(_p)
        else:
            _incompat[_p] = _why
            st.sidebar.error(f"⛔ {_p} — Not Compatible: {_why}")
    precisions = _compatible

    # Memory guard — warn per precision, mark which will be skipped
    # Use conservative estimates: batch=32 for Batch Throughput; long context = 1024 tokens
    if precisions and selected_model and "/" in selected_model:
        _chk_batch = 32 if scenario == "Batch Throughput" else 1
        _chk_ctx   = 1024 if scenario == "Long Context" else 256
        _mem_check = check_precisions(selected_model, precisions,
                                      batch_size=_chk_batch, context_len=_chk_ctx)
        _blocked   = [p for p, s in _mem_check.items() if not s["fits"]]
        _allowed   = [p for p, s in _mem_check.items() if s["fits"]]
        for _p, _s in _mem_check.items():
            _est = _s["est_gb"]
            if _s["fits"]:
                st.sidebar.caption(f"✅ {_p} — ~{_est:.0f} GB  (usable: {GB10_USABLE_GB:.0f} GB)")
            else:
                st.sidebar.error(f"🚫 {_p} — ~{_est:.0f} GB exceeds {GB10_USABLE_GB:.0f} GB usable. Will be skipped.")
    else:
        _blocked = []
        _allowed = list(precisions)

    num_runs = st.sidebar.slider("Runs per precision", 2, 8, 3)
    if scenario == "Batch Throughput":
        batch_size = st.sidebar.number_input(
            "Batch size", min_value=1, max_value=2048, value=32, step=16,
            help="Dell Pro Max GB10 has 128 GB unified memory (~100 GB usable) — push past 32 to stress memory"
        )
    elif scenario == "Batch Limit Sweep":
        sweep_start = st.sidebar.number_input("Start batch", min_value=1, max_value=256, value=1, step=1)
        sweep_max   = st.sidebar.number_input("Max batch (or until OOM)", min_value=2, max_value=2048, value=512, step=32)
        sweep_prec  = st.sidebar.selectbox("Precision for sweep", precision_opts, index=1)
        batch_size = sweep_start
    else:
        batch_size = 1
    num_users = 1
    context_length = "long" if scenario == "Long Context" else "short"
    gen_local_tp = st.sidebar.checkbox(
        "📝 On-device talking points", value=True,
        help="After the run, have THIS model generate its own results summary "
             "locally on the Dell Pro Max GB10 (no cloud) and show it in Results.",
    )

elif scenario == "Multi-User Concurrency":
    num_users = st.sidebar.select_slider("Concurrent users", [1, 2, 4, 8, 16], value=4)
    model_size_gb = st.sidebar.slider("Model size (GB each)", 1.0, 40.0, 7.0, step=1.0)
    precision_opts = ["FP32", "FP16", "BF16"]
    precisions = st.sidebar.multiselect("Precisions", precision_opts, default=["FP16"])
    num_runs = 3
    batch_size = 1
    selected_model = f"{num_users}-user concurrency"
    context_length = "short"
    _blocked, _allowed = [], list(precisions)
    # Concurrency check: total = model_size_gb × num_users × bytes_per_param overhead
    for _p in precisions:
        _total = model_size_gb * num_users * 1.12
        if _total > GB10_USABLE_GB:
            st.sidebar.error(f"🚫 {_p} — {num_users} × {model_size_gb:.0f} GB = ~{_total:.0f} GB exceeds {GB10_USABLE_GB:.0f} GB usable. Will be skipped.")
            _blocked.append(_p)
        else:
            st.sidebar.caption(f"✅ {_p} — ~{_total:.0f} GB total  (usable: {GB10_USABLE_GB:.0f} GB)")
    _allowed = [p for p in precisions if p not in _blocked]

elif scenario == "Memory Pressure":
    fill_pct = st.sidebar.slider("Fill target (%)", 50, 98, 85)
    precision_opts = ["FP32", "FP16", "BF16", "FP64"]
    precisions = st.sidebar.multiselect("Precisions", precision_opts, default=["FP32", "FP16"])
    num_runs = st.sidebar.slider("Runs", 2, 6, 3)
    batch_size = 1
    num_users = 1
    selected_model = f"Memory Fill {fill_pct}%"
    context_length = "n/a"
    _blocked, _allowed = [], list(precisions)  # memory pressure intentionally fills; no pre-check

elif scenario == "Vision Throughput":
    vision_models = [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14",
        "google/vit-base-patch16-224",
        "facebook/dino-vits16",
        "microsoft/resnet-50",
        "google/efficientnet-b4",
    ]
    selected_model = st.sidebar.selectbox("Model", vision_models)
    precision_opts = ["FP32", "FP16", "BF16", "INT8", "FP4", "NVFP4"]
    precisions = st.sidebar.multiselect("Precisions", precision_opts, default=["FP32", "FP16"])
    batch_size = st.sidebar.number_input(
        "Batch size", min_value=1, max_value=2048, value=8, step=8,
        help="Large batches stress VRAM — try 128+ on the Dell Pro Max GB10"
    )
    num_runs = st.sidebar.slider("Runs", 2, 8, 3)
    num_users = 1
    context_length = "n/a"
    # Vision models are tiny — no pre-check needed, all precisions pass
    _blocked, _allowed = [], list(precisions)

else:  # HPC / Quant Analysis
    hpc_tests = ["MatMul Benchmark", "Bandwidth Test", "LOB Bandwidth", "Reduction Ops",
                 "Fill Memory (60%)", "Dual Model Serving",
                 "Black-Scholes Options", "Monte Carlo VaR"]
    selected_model = st.sidebar.selectbox("Test", hpc_tests)
    # TF32 = FP32-storage tensor-core matmul mode; only differs from FP32 on the
    # MatMul test (tensor cores), identical to FP32 on bandwidth/reduction tests.
    precision_opts = ["FP64", "FP32", "TF32", "FP16", "BF16"]
    precisions = st.sidebar.multiselect("Precisions", precision_opts, default=["FP32", "TF32"])
    num_runs = st.sidebar.slider("Runs", 2, 8, 3)
    hpc_target_gb = st.sidebar.slider(
        "GPU memory target (GB)", 4, 80, 32, step=4,
        help="Sizes each test to consume ~this much of the Dell Pro Max GB10's 128 GB unified memory, "
             "so it saturates the real resource (bandwidth / compute / capacity). "
             "The Results show which one was the bottleneck.",
    )
    batch_size = 1
    num_users = 1
    context_length = "n/a"
    _blocked, _allowed = [], list(precisions)

if not precisions:
    st.sidebar.warning("Select at least one precision")
elif _allowed and len(_blocked) > 0:
    st.sidebar.warning(f"⚠ {len(_blocked)} precision(s) will be skipped due to memory constraints. {len(_allowed)} will run.")
elif not _allowed and precisions:
    st.sidebar.error("🚫 All selected precisions exceed memory limits. Choose lower-precision options.")

st.sidebar.divider()
run_benchmark = st.sidebar.button("🚀 Run Benchmark", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="dell-header">
  <div>
    <div class="dell-logo"><strong>DELL</strong> &nbsp;technologies</div>
    <div class="dell-title">Dell Pro Max GB10 Demo Suite</div>
    <div class="dell-subtitle">Blackwell · Dell Pro Max GB10 · aarch64 &nbsp;|&nbsp; AI Workload Benchmarking</div>
  </div>
  <div class="dell-badge">
    Dell Pro Max GB10<br>
    <span style="font-size:1.2em;font-weight:700;">Blackwell</span><br>
    128 GB Unified Memory
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Scenario", scenario)
c2.metric("Model", (selected_model[:22] + "…") if len(selected_model) > 24 else selected_model)
c3.metric("Precisions", len(precisions))
st.divider()

# ---------------------------------------------------------------------------
# Always-on narrator lifecycle
# ---------------------------------------------------------------------------
# The narrator (see NARRATOR_MODEL/NARRATOR_PRECISION in on_device_ai.py) loads
# ON DEMAND — only when Deep dive / talking points is clicked — then stays resident
# in session_state so later calls are instant. It is unloaded + GPU flushed before a
# benchmark run and NOT reloaded afterwards; the next Deep dive click brings it back.

def ensure_narrator(show_spinner: bool = True):
    """Load the narrator on first use; keep it resident in session_state thereafter."""
    n = st.session_state.get("narrator")
    if n is not None and getattr(n, "ready", False) and getattr(n, "inf", None) is not None:
        return n
    ok, _ = narrator_available()
    if not ok:
        st.session_state.narrator = None
        return None
    if show_spinner:
        with st.spinner(f"🟢 Bringing up on-device narrator ({NARRATOR_MODEL.split('/')[-1]} @ {NARRATOR_PRECISION}) on the Dell Pro Max GB10…"):
            n = make_narrator()
    else:
        n = make_narrator()
    st.session_state.narrator = n if getattr(n, "ready", False) else None
    return st.session_state.narrator


def unload_narrator_for_benchmark():
    """Unload the resident narrator and flush GPU memory before benchmarking."""
    n = st.session_state.get("narrator")
    if n is not None:
        n.unload()
    st.session_state.narrator = None
    free_cuda_memory()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_settings, tab_benchmark, tab_results, tab_tco = st.tabs(
    ["Settings", "Benchmark", "Results", "TCO Analysis"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Settings
# ---------------------------------------------------------------------------
with tab_settings:
    st.subheader("Benchmark Configuration")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Selected Configuration**")
        cfg = {
            "scenario": scenario,
            "model": selected_model,
            "precisions": precisions,
            "num_runs": num_runs,
            "batch_size": batch_size,
        }
        if scenario == "Multi-User Concurrency":
            cfg["num_users"] = num_users
        st.json(cfg)
    with c2:
        st.write("**What this scenario measures**")
        guides = {
            "Quick Inference": "TTFT + decode latency at batch=1. Primary indicator: time to first token (user experience).",
            "Long Context": "Prefill cost scales with context length. Exposes memory-bandwidth limit on long sequences.",
            "Batch Throughput": "Peak tokens/sec at large batch. Dell Pro Max GB10 has 128 GB unified memory — push batch into the hundreds to stress VRAM.",
            "Batch Limit Sweep": "Auto-escalates batch geometrically (1→2→4→8…→OOM). Plots tokens/sec and VRAM% to find the exact memory capacity ceiling.",
            "Multi-User Concurrency": "How many sessions fit in 128 GB unified memory. Exposes memory capacity limit.",
            "Memory Pressure": "Sustained bandwidth at high memory fill. Exposes bandwidth ceiling.",
            "Vision Throughput": "Images/sec across precisions. TTFT = encode latency.",
            "HPC / Quant Analysis": "Raw TFLOPS (MatMul), GB/s (Bandwidth/LOB), reduction throughput.",
        }
        st.info(guides.get(scenario, ""))
        st.markdown("""
**Precision Guide**
- **FP32 / FP64**: Baseline / scientific precision
- **FP16 / BF16**: ~2× faster, same VRAM footprint
- **INT8**: ~2–3× faster, ~2× VRAM reduction
- **FP4 / NVFP4**: ~4–8× faster, ~4× VRAM reduction (Blackwell hardware-accelerated)
        """)

    # -----------------------------------------------------------------------
    # Detailed model / test explanation (curated, always available) + Claude deep-dive
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("#### 📖 About this model / test")
    _minfo = lookup_model_info(selected_model)
    if _minfo:
        st.markdown(f"**{selected_model}** — {_minfo['headline']}")
        st.markdown(_minfo["body"])
    else:
        st.caption(
            f"**{selected_model}** is a synthetic workload (sized tensors), not a specific "
            "published model — it stresses memory capacity / bandwidth / concurrency directly."
        )
    _ai_ok, _ai_reason = narrator_available()
    if st.button("🔍 Deep dive (on-device AI)", key="ai_model_explain",
                 disabled=not _ai_ok,
                 help=None if _ai_ok else _ai_reason):
        _narr = ensure_narrator()   # resident — already loaded, stays loaded
        if _narr and _narr.ready:
            st.write_stream(_narr.explain_model(
                selected_model, scenario, _minfo["body"] if _minfo else ""))
        else:
            st.warning(_narr.reason if _narr else _ai_reason)
    if not _ai_ok:
        st.caption(f"💡 {_ai_reason}")

# ---------------------------------------------------------------------------
# TAB 2 — Benchmark
# ---------------------------------------------------------------------------
with tab_benchmark:
    st.subheader("Run Benchmark")

    if run_benchmark:
        # Free the resident narrator and flush GPU memory so the benchmark gets a
        # clean, uncontended Dell Pro Max GB10 to test against.
        _had_narrator = st.session_state.get("narrator") is not None
        unload_narrator_for_benchmark()
        if _had_narrator:
            st.caption("🧹 Narrator unloaded · GPU memory flushed for a clean benchmark")

        results = []
        local_summary = None   # on-device talking points (LLM scenarios only)

        if scenario == "Batch Limit Sweep":
            inf = LLMInference(selected_model, sweep_prec)
            sweep_results = []
            if not inf.load_model():
                st.error(f"Failed to load {selected_model} at {sweep_prec}")
            else:
                batch = sweep_start
                # Geometric steps: 1,2,4,8,16,32,64,128,256… then linear above 256
                def _next_batch(b):
                    if b < 64:
                        return b * 2
                    if b < 256:
                        return b + 64
                    return b + 128

                progress = st.progress(0.0, text=f"Sweep starting at batch={batch}…")
                oom_hit = False
                step = 0
                total_steps_est = 12  # rough

                while batch <= sweep_max and not oom_hit:
                    progress.progress(min(step / total_steps_est, 0.95),
                                      text=f"batch={batch} @ {sweep_prec}…")
                    m = inf.benchmark(num_runs=2, batch_size=batch, context_length="short")
                    sweep_results.append((batch, m))
                    if m.error and "OOM" in (m.error or ""):
                        st.warning(f"⚡ OOM at batch={batch} — limit found!")
                        oom_hit = True
                    elif m.error:
                        st.error(f"Error at batch={batch}: {m.error}")
                        break
                    else:
                        tps = m.business_output.get("tokens_per_sec", 0)
                        mem_pct = m.operational_condition.get("mem_pct", 0)
                        st.success(f"✓ batch={batch}: {tps:.0f} tok/s | {mem_pct:.0f}% VRAM | {m.primary_bottleneck}")
                    batch = _next_batch(batch)
                    step += 1

                progress.empty()
                # On-device talking points from the peak result, before unloading.
                _ok_runs = [(b, m) for b, m in sweep_results if not m.error]
                if gen_local_tp and _ok_runs:
                    _pb, _pm = max(_ok_runs, key=lambda x: x[1].business_output.get("tokens_per_sec", 0))
                    _ptps = _pm.business_output.get("tokens_per_sec", 0)
                    _dp, _dpn = designed_precision(selected_model)
                    with st.spinner("📝 Dell Pro Max GB10 is writing its own results summary…"):
                        _txt = inf.generate_text(_local_tp_prompt(
                            selected_model, sweep_prec, _ptps,
                            _pm.business_output.get("ttft_ms", _pm.latency_ms),
                            _pm.operational_condition.get("mem_pct", 0),
                            _pm.primary_bottleneck, _dp, _dpn))
                    if _txt:
                        local_summary = {"text": _txt, "model": selected_model,
                                         "precision": sweep_prec, "tps": _ptps}
                inf.unload()

            st.session_state.results = {
                "scenario": scenario,
                "model": selected_model,
                "metrics": [m for _, m in sweep_results if not m.error],
                "sweep_data": sweep_results,
                "sweep_prec": sweep_prec,
                "local_summary": local_summary,
            }
            if sweep_results:
                st.success("✅ Sweep complete — see Results tab")

        elif scenario in ("Quick Inference", "Long Context", "Batch Throughput"):
            for precision in _allowed:
                with st.spinner(f"⏳ {selected_model} @ {precision} — loading + generate…"):
                    inf = LLMInference(selected_model, precision)
                    try:
                        if inf.load_model():
                            m = inf.benchmark(
                                num_runs=num_runs,
                                batch_size=batch_size,
                                context_length=context_length,
                                num_users=num_users,
                            )
                            results.append(m)
                            if m.error:
                                st.error(f"✗ {precision}: {m.error}")
                            else:
                                ttft = m.business_output.get("ttft_ms", m.latency_ms)
                                tps = m.business_output.get("tokens_per_sec", 0)
                                _plabel = (f"{precision} → {m.effective_precision}"
                                           if getattr(m, "precision_note", "") and
                                           m.effective_precision != precision else precision)
                                st.success(f"✓ {_plabel}: TTFT {ttft:.0f} ms | {tps:.0f} tok/s | {m.primary_bottleneck}")
                                if getattr(m, "precision_note", ""):
                                    st.caption(f"ⓘ {m.precision_note}")
                                # On-device talking points — let this model narrate its own
                                # run on the Dell Pro Max GB10 before we unload it (first success only).
                                if gen_local_tp and local_summary is None:
                                    _dp, _dpn = designed_precision(selected_model)
                                    with st.spinner("📝 Dell Pro Max GB10 is writing its own results summary…"):
                                        _txt = inf.generate_text(_local_tp_prompt(
                                            selected_model, precision, tps,
                                            ttft, m.operational_condition.get("mem_pct", 0),
                                            m.primary_bottleneck, _dp, _dpn))
                                    if _txt:
                                        local_summary = {"text": _txt, "model": selected_model,
                                                         "precision": precision, "tps": tps}
                        else:
                            st.error(f"✗ {precision}: failed to load")
                    except Exception as e:
                        st.error(f"Error @ {precision}: {e}")
                    finally:
                        inf.unload()

        elif scenario == "Vision Throughput":
            for precision in _allowed:
                with st.spinner(f"⏳ {selected_model} @ {precision}…"):
                    inf = VisionModelInference(selected_model, precision)
                    try:
                        if inf.load_model():
                            m = inf.benchmark(num_runs=num_runs, batch_size=batch_size)
                            results.append(m)
                            if m.error:
                                st.error(f"✗ {precision}: {m.error}")
                            else:
                                ips = m.business_output.get("images_per_sec", 0)
                                st.success(f"✓ {precision}: {ips:.1f} img/s | {m.primary_bottleneck}")
                        else:
                            st.error(f"✗ {precision}: failed to load")
                    except Exception as e:
                        st.error(f"Error @ {precision}: {e}")
                    finally:
                        inf.unload()

        elif scenario == "Multi-User Concurrency":
            for precision in _allowed:
                with st.spinner(f"⏳ {num_users} concurrent × {model_size_gb:.0f}GB @ {precision}…"):
                    try:
                        m = MemoryStress.concurrent_models_benchmark(
                            num_models=num_users,
                            model_size_gb=model_size_gb,
                            precision=precision,
                        )
                        results.append(m)
                        spill = m.business_output.get("spills", False)
                        loaded = m.business_output.get("models_loaded", 0)
                        flag = "⚠ SPILL" if spill else "✓ fits"
                        st.success(f"{flag} — {loaded}/{num_users} loaded @ {precision} | {m.primary_bottleneck}")
                    except Exception as e:
                        st.error(f"Error @ {precision}: {e}")

        elif scenario == "Memory Pressure":
            for precision in _allowed:
                with st.spinner(f"⏳ Fill {fill_pct}% @ {precision}…"):
                    try:
                        m = MemoryStress.fill_memory_benchmark(
                            target_percent=fill_pct / 100,
                            precision=precision,
                            num_runs=num_runs,
                        )
                        results.append(m)
                        if m.error:
                            st.error(f"✗ {precision}: {m.error}")
                        else:
                            bw = m.business_output.get("bandwidth_gbs", m.throughput_samples_per_sec)
                            st.success(f"✓ {precision}: {bw:.0f} GB/s @ {fill_pct}% fill | {m.primary_bottleneck}")
                    except Exception as e:
                        st.error(f"Error @ {precision}: {e}")

        else:  # HPC
            with st.spinner(f"⏳ {selected_model} @ ~{hpc_target_gb} GB target…"):
                try:
                    for precision in _allowed:
                        # Size each workload to consume ~hpc_target_gb of unified memory
                        _bpe   = {"FP64": 8, "FP32": 4, "FP16": 2, "BF16": 2}.get(precision, 4)
                        _tgt_b = hpc_target_gb * (1024 ** 3)

                        if selected_model == "MatMul Benchmark":
                            # 3 n×n matrices (A,B,C); cap size to keep O(n³) runtime interactive
                            _n = int((_tgt_b / (3 * _bpe)) ** 0.5)
                            _n = max(4096, min(_n, 16384)) // 256 * 256
                            m = HPCBenchmark.matmul_benchmark(matrix_size=_n, precision=precision, num_runs=num_runs)
                        elif selected_model == "Bandwidth Test":
                            # tensor + one temp ≈ 2× → size tensor to half the target
                            _mb = max(512, int(hpc_target_gb * 1024 * 0.5))
                            m = HPCBenchmark.bandwidth_benchmark(size_mb=_mb, precision=precision, num_runs=num_runs)
                        elif selected_model == "LOB Bandwidth":
                            # order_book(levels×8) + gathered/delta(ops×8) ≈ 24×bytes per level
                            _levels = max(1_000_000, int(_tgt_b / (_bpe * 8 * 3)))
                            m = HPCBenchmark.lob_benchmark(num_price_levels=_levels,
                                                           num_operations=2 * _levels,
                                                           precision=precision, num_runs=num_runs)
                        elif selected_model == "Reduction Ops":
                            _vec = max(64_000_000, int(_tgt_b / _bpe * 0.85))
                            m = HPCBenchmark.reduction_benchmark(vector_size=_vec, precision=precision, num_runs=num_runs)
                        elif selected_model == "Fill Memory (60%)":
                            m = MemoryStress.fill_memory_benchmark(target_percent=0.60, precision=precision, num_runs=num_runs)
                        elif selected_model == "Black-Scholes Options":
                            # ~11 working arrays of n elements
                            _opts = max(1_000_000, int(_tgt_b / (_bpe * 11)))
                            m = HPCBenchmark.black_scholes_benchmark(num_options=_opts, precision=precision, num_runs=num_runs)
                        elif selected_model == "Monte Carlo VaR":
                            # Z + log_ret ≈ 2 × paths × assets × days
                            _paths = max(500_000, int(_tgt_b / (_bpe * 2 * 100 * 10)))
                            m = HPCBenchmark.montecarlo_var_benchmark(num_paths=_paths, precision=precision, num_runs=num_runs)
                        else:  # Dual Model Serving
                            m = MemoryStress.concurrent_models_benchmark(num_models=2, model_size_gb=10.0, precision=precision)
                        results.append(m)
                        if m.error:
                            st.error(f"✗ {precision}: {m.error}")
                        else:
                            _bo = m.business_output
                            _tf = _bo.get("tflops")
                            _bw = _bo.get("bandwidth_gbs")
                            _gf = _bo.get("gflops")
                            if _tf:
                                _metric = f"{_tf:,.0f} TFLOPS"
                            elif _bw:
                                _metric = f"{_bw:,.0f} GB/s"
                            elif _gf:
                                _metric = f"{_gf:,.0f} GFLOPS"
                            else:
                                _metric = f"{m.latency_ms:,.1f} ms"
                            _memp = m.operational_condition.get("mem_pct", 0)
                            st.success(f"✓ {precision}: {_metric} · {_memp:.0f}% VRAM · 🔻 bottleneck: **{m.primary_bottleneck}**")
                except Exception as e:
                    st.error(f"Benchmark error: {e}")

        # Batch Limit Sweep stores its own richer result dict (with sweep_data)
        # inside its branch above — don't clobber it with the generic list.
        if scenario != "Batch Limit Sweep":
            st.session_state.results = {
                "scenario": scenario,
                "model": selected_model,
                "metrics": results,
                "local_summary": local_summary,
                "incompatible": dict(_incompat),
            }
            if results:
                st.success("✅ Benchmark complete — see Results tab")
            if _incompat:
                st.warning("⛔ Not Compatible (not run): "
                           + " · ".join(f"{p} ({why})" for p, why in _incompat.items()))

        # Authoritative post-run cleanup: every benchmark helper has returned by
        # now, so its local tensors are unreferenced — return that VRAM to the
        # driver so top/nvidia-smi reflect a clean slate before the next run.
        free_cuda_memory()
        _free = MemoryStress.get_available_memory_mb()
        if _free > 0:
            st.caption(f"🧹 GPU memory released — {_free/1024:.1f} GB free")

    else:
        st.info("👈 Configure in sidebar then click **Run Benchmark**")

# ---------------------------------------------------------------------------
# TAB 3 — Results
# ---------------------------------------------------------------------------
with tab_results:
    st.subheader("Benchmark Results")

    # Precisions the model's checkpoint can't run — shown as Not Compatible,
    # never executed, and excluded from the metrics/TCO below.
    _incompat_done = (st.session_state.results or {}).get("incompatible") or {}
    if _incompat_done:
        st.warning("⛔ **Not Compatible** (excluded from results & TCO): "
                   + " · ".join(f"**{p}** — {why}" for p, why in _incompat_done.items()))

    if not (st.session_state.results and st.session_state.results.get("metrics")):
        st.info("No results yet — run a benchmark first.")
        st.stop()

    rdata = st.session_state.results
    all_metrics: list[BenchmarkMetrics] = [m for m in rdata["metrics"] if not m.error]
    err_metrics: list[BenchmarkMetrics] = [m for m in rdata["metrics"] if m.error]
    # Always present results in precision order: lowest bit-width → highest.
    all_metrics.sort(key=lambda m: _precision_rank(m.precision))
    err_metrics.sort(key=lambda m: _precision_rank(m.precision))

    if err_metrics:
        for m in err_metrics:
            st.error(f"✗ {m.precision}: {m.error}")

    if not all_metrics:
        st.warning("All runs produced errors.")
        st.stop()

    st.caption(f"**Scenario:** {rdata['scenario']}   |   **Model:** {rdata['model']}")

    # On-device talking points — generated by the benchmarked model itself, on the Dell Pro Max GB10
    _local = rdata.get("local_summary")
    if _local and _local.get("text"):
        st.markdown(
            f"#### 🟢 On-Device Talking Points "
            f"<span style='font-size:0.7em;color:#76B900;'>— generated locally by "
            f"{_local['model'].split('/')[-1]} @ {_local['precision']} on the Dell Pro Max GB10 "
            f"({_local['tps']:.0f} tok/s), no cloud</span>",
            unsafe_allow_html=True,
        )
        # Precision context: what it's designed for + what ran fastest in this run
        _dp, _dpn = designed_precision(_local["model"])
        _fast = max(all_metrics, key=lambda m: getattr(m, "tokens_per_sec", 0) or 0, default=None)
        _pline = f"**🎯 Designed for:** {_dp or '—'}"
        if _dpn:
            _pline += f" ({_dpn})"
        if _fast is not None and (getattr(_fast, "tokens_per_sec", 0) or 0) > 0:
            _pline += f"  ·  **⚡ Fastest in this run:** {_fast.precision} ({_fast.tokens_per_sec:.0f} tok/s)"
        st.markdown(_pline)
        st.success(_local["text"])
    else:
        # Scenarios that can't self-narrate (vision / HPC / memory): run the on-device
        # narrator model on the Dell Pro Max GB10 to generate the talking points locally.
        with st.expander("🟢 On-Device AI Talking Points", expanded=False):
            _ai_ok, _ai_reason = narrator_available()
            st.caption("Generate sales talking points from these results — locally on the Dell Pro Max GB10, no cloud."
                       if _ai_ok else f"⚠ {_ai_reason}")
            if st.button("Generate on-device", key="ai_results", disabled=not _ai_ok):
                _narr = ensure_narrator()   # resident
                if _narr and _narr.ready:
                    st.write_stream(_narr.results_summary(rdata))
                else:
                    st.warning(_narr.reason if _narr else _ai_reason)

    # -----------------------------------------------------------------------
    # BATCH LIMIT SWEEP CHART (shown instead of normal layout when applicable)
    # -----------------------------------------------------------------------
    if rdata.get("sweep_data"):
        sweep_data = rdata["sweep_data"]
        sweep_prec = rdata.get("sweep_prec", "")
        batches, tps_vals, mem_pcts, oom_batch = [], [], [], None

        for b, m in sweep_data:
            if m.error and "OOM" in (m.error or ""):
                oom_batch = b
                break
            tps = m.business_output.get("tokens_per_sec", 0)
            mem = m.operational_condition.get("mem_pct", 0)
            batches.append(b)
            tps_vals.append(tps)
            mem_pcts.append(mem)

        st.markdown(f"#### Batch Throughput Sweep — {sweep_prec}")
        if oom_batch:
            st.error(f"💥 OOM at batch={oom_batch} — that's the Dell Pro Max GB10 limit for {rdata['model']} @ {sweep_prec}")

        if batches:
            fig_sweep = go.Figure()
            fig_sweep.add_trace(go.Scatter(
                x=batches, y=tps_vals, name="Tokens/sec",
                mode="lines+markers", line=dict(color="#007DB8", width=2),
                marker=dict(size=8),
            ))
            if oom_batch:
                fig_sweep.add_vline(
                    x=oom_batch, line_dash="dash", line_color="#dc3545",
                    annotation_text=f"OOM @ {oom_batch}", annotation_position="top right",
                )
            fig_sweep.update_layout(
                height=360, xaxis_title="Batch Size", yaxis_title="Tokens / sec",
                showlegend=False, hovermode="x unified",
            )
            st.plotly_chart(fig_sweep, use_container_width=True)

            fig_mem = go.Figure()
            fig_mem.add_trace(go.Scatter(
                x=batches, y=mem_pcts, name="VRAM %",
                mode="lines+markers", line=dict(color="#E87722", width=2),
                fill="tozeroy", fillcolor="rgba(232,119,34,0.12)",
            ))
            fig_mem.add_hline(y=95, line_dash="dot", line_color="#dc3545",
                              annotation_text="95% VRAM ceiling")
            if oom_batch:
                fig_mem.add_vline(x=oom_batch, line_dash="dash", line_color="#dc3545")
            fig_mem.update_layout(
                height=280, xaxis_title="Batch Size", yaxis_title="VRAM %",
                showlegend=False,
            )
            st.plotly_chart(fig_mem, use_container_width=True)

            # Key finding callout
            if tps_vals:
                peak_tps = max(tps_vals)
                peak_batch = batches[tps_vals.index(peak_tps)]
                c1, c2, c3 = st.columns(3)
                c1.metric("Peak Tokens/sec", f"{peak_tps:.0f}", f"at batch={peak_batch}")
                c2.metric("Last VRAM%", f"{mem_pcts[-1]:.0f}%", f"batch={batches[-1]}")
                if oom_batch:
                    c3.metric("OOM Boundary", f"batch={oom_batch}", "Memory Capacity limit")

        st.divider()

    # -----------------------------------------------------------------------
    # 1. BUSINESS SUMMARY CARD
    # -----------------------------------------------------------------------
    st.markdown("#### Business Summary")

    # Aggregate across precisions — show best result per metric
    best = all_metrics[0]
    for m in all_metrics[1:]:
        if m.tokens_per_sec > best.tokens_per_sec:
            best = m

    ttft    = best.business_output.get("ttft_ms", best.latency_ms)
    tps     = best.business_output.get("tokens_per_sec", best.tokens_per_sec)
    ips     = best.business_output.get("images_per_sec", 0)
    cpm     = best.business_output.get("cost_per_mtok")
    mcs     = best.business_output.get("max_concurrent_sessions")
    p95     = best.business_output.get("p95_ms", best.latency_ms * 1.15)

    bc1, bc2, bc3, bc4, bc5 = st.columns(5)
    with bc1:
        if ttft and ttft > 0:
            st.metric("⚡ Time to First Token", f"{ttft:.0f} ms",
                      help="Prefill latency — how fast user sees first output token")
    with bc2:
        if tps and tps > 0:
            st.metric("🔥 Tokens / sec", f"{tps:.0f}",
                      help="Sustained decode throughput at best precision")
        elif ips and ips > 0:
            st.metric("🖼 Images / sec", f"{ips:.1f}",
                      help="Image encode throughput at best precision")
    with bc3:
        if cpm:
            st.metric("💰 Cost / MTok", f"${cpm:.4f}",
                      help="Equivalent H100 cloud cost per million output tokens")
    with bc4:
        if mcs:
            st.metric("👥 Max Sessions", str(mcs),
                      help="Concurrent sessions that fit in 128 GB unified memory")
    with bc5:
        if p95:
            st.metric("📊 p95 Latency", f"{p95:.0f} ms",
                      help="95th-percentile latency across all runs")

    # -----------------------------------------------------------------------
    # 1b. PERFORMANCE SCORECARD  (RAG badges) + SERVING METRICS
    # -----------------------------------------------------------------------
    def _rag(label, value_str, color, note=""):
        _bg = {"green": "#198754", "yellow": "#e6a817", "red": "#dc3545"}.get(color, "#6c757d")
        _tc = "#fff" if color != "yellow" else "#1d1d1b"
        return (
            f"<div style='display:inline-flex;flex-direction:column;align-items:center;"
            f"background:{_bg};color:{_tc};border-radius:6px;padding:10px 18px;margin:3px;"
            f"min-width:130px;box-shadow:0 1px 5px rgba(0,0,0,0.18);'>"
            f"<span style='font-size:0.62em;font-weight:700;letter-spacing:0.09em;"
            f"text-transform:uppercase;opacity:0.85;'>{label}</span>"
            f"<span style='font-size:1.15em;font-weight:800;margin:3px 0;'>{value_str}</span>"
            f"<span style='font-size:0.6em;opacity:0.8;'>{note}</span></div>"
        )

    _bw_util  = best.business_output.get("bw_util_pct", 0) or 0
    _tps_sc   = best.business_output.get("tokens_per_sec", best.tokens_per_sec) or 0
    _ips_sc   = best.business_output.get("images_per_sec", 0) or 0
    _tput_sc  = _tps_sc or _ips_sc
    _prec_sc  = best.precision.upper()

    # Mem BW Utilization
    _bw_color = "green" if _bw_util >= 55 else ("yellow" if _bw_util >= 25 else "red")
    # Throughput
    _tput_color = "green" if _tput_sc >= 50 else ("yellow" if _tput_sc >= 10 else "red")
    # Scaling efficiency — single GPU run = 100 %
    _eff_color = "green"
    # Quality (precision-based)
    _qual_color = "green" if any(p in _prec_sc for p in ("FP32","FP16","BF16")) else \
                  "yellow" if "INT8" in _prec_sc else "yellow"
    _qual_note  = "Full precision" if any(p in _prec_sc for p in ("FP32","FP16","BF16")) else \
                  "~98% of FP16" if "INT8" in _prec_sc else "HW-accel quant"
    # Precision appropriateness
    _prec_color = "green" if _prec_sc in ("BF16","FP16","NVFP4","FP4") else \
                  "yellow" if _prec_sc == "INT8" else "yellow"
    _prec_note  = "Optimal" if _prec_sc in ("BF16","FP16") else \
                  "Blackwell native" if "NVFP4" in _prec_sc or _prec_sc == "FP4" else \
                  "Good trade-off" if _prec_sc == "INT8" else "Use BF16 instead"

    st.markdown("#### Performance Scorecard")
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:2px;margin-bottom:14px;'>"
        + _rag("Mem BW Util",      f"{_bw_util:.1f}%",   _bw_color,   "vs Dell Pro Max GB10 273 GB/s")
        + _rag("Throughput",       f"{_tput_sc:.0f}",    _tput_color, "tok/s" if _tps_sc > 0 else "img/s")
        + _rag("Scale Eff.",       "100%",               _eff_color,  "single GPU")
        + _rag("Final Quality",    _prec_sc,             _qual_color, _qual_note)
        + _rag("Precision",        _prec_sc,             _prec_color, _prec_note)
        + "</div>",
        unsafe_allow_html=True,
    )

    _qps  = best.business_output.get("qps", 0) or 0
    _ttft = best.business_output.get("ttft_ms", best.latency_ms) or 0
    _tpot = best.business_output.get("tpot_ms", 0) or 0
    _itl  = best.business_output.get("itl_ms", 0) or 0
    _p99  = best.business_output.get("p99_ms_est", 0) or 0
    _bw_abs = best.business_output.get("bw_gbs_used", 0) or 0

    _sm = st.columns(6)
    if _qps  > 0: _sm[0].metric("QPS",        f"{_qps:.2f}",       help="Queries per second  (batch ÷ end-to-end latency)")
    if _ttft > 0: _sm[1].metric("TTFT",        f"{_ttft:.0f} ms",   help="Time to first token — prefill latency")
    if _tpot > 0: _sm[2].metric("TPOT",        f"{_tpot:.1f} ms",   help="Time per output token during decode")
    if _itl  > 0: _sm[3].metric("ITL",         f"{_itl:.1f} ms",    help="Inter-token latency (= TPOT single-stream)")
    if _p99  > 0: _sm[4].metric("P99 (est.)",  f"{_p99:.0f} ms",    help="99th-percentile latency estimated from run spread")
    if _bw_abs>0: _sm[5].metric("BW Used",     f"{_bw_abs:.0f} GB/s", help="Estimated memory bandwidth consumed during decode")

    st.divider()

    # -----------------------------------------------------------------------
    # 2. BOTTLENECK BADGES  +  per-precision row
    # -----------------------------------------------------------------------
    st.markdown("#### Per-Precision Results")

    for m in all_metrics:
        badge = bottleneck_badge_html(m.primary_bottleneck) if m.primary_bottleneck else ""
        phase_label = f"<span style='color:#aaa;font-size:0.82em;margin-left:10px;'>{m.workload_phase}</span>" if m.workload_phase else ""
        # When the precision actually run differs from the one requested (e.g. FP16 on a
        # BF16-native model runs as BF16), show it honestly as "FP16 → BF16" with the reason.
        _eff = getattr(m, "effective_precision", m.precision)
        _note = getattr(m, "precision_note", "")
        _prec_label = f"{m.precision} → {_eff}" if (_note and _eff != m.precision) else m.precision
        _note_html = (f"<span style='color:#e0a020;font-size:0.8em;margin-left:10px;'>ⓘ {_note}</span>"
                      if _note else "")
        st.markdown(
            f"**{_prec_label}** &nbsp; {badge} {phase_label}{_note_html}",
            unsafe_allow_html=True,
        )
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Latency", f"{m.latency_ms:.1f} ms")
        r2.metric("Peak VRAM", f"{m.peak_memory_mb:.0f} MB")
        tps_val  = m.business_output.get("tokens_per_sec", m.tokens_per_sec)
        ips_val  = m.business_output.get("images_per_sec", 0)
        bw_val   = m.business_output.get("bandwidth_gbs", 0)
        opts_val = m.business_output.get("options_per_sec_M", 0)
        paths_val= m.business_output.get("paths_per_sec_M", 0)
        if tps_val and tps_val > 0:
            r3.metric("Tokens/sec", f"{tps_val:.0f}")
        elif ips_val and ips_val > 0:
            r3.metric("Images/sec", f"{ips_val:.1f}")
        elif opts_val and opts_val > 0:
            r3.metric("M Options/sec", f"{opts_val:.2f}")
        elif paths_val and paths_val > 0:
            r3.metric("M Paths/sec", f"{paths_val:.2f}")
        elif bw_val and bw_val > 0:
            r3.metric("Bandwidth", f"{bw_val:.0f} GB/s")
        load_ms = m.business_output.get("load_ms", 0)
        if load_ms:
            r4.metric("Load time", f"{load_ms:.0f} ms")

    st.divider()

    # -----------------------------------------------------------------------
    # 3. PHASE WATERFALL  (horizontal stacked bar)
    # -----------------------------------------------------------------------
    st.markdown("#### Phase Breakdown")

    has_phases = any(
        m.business_output.get("load_ms", 0) > 0 or
        m.business_output.get("ttft_ms", 0) > 0 or
        m.business_output.get("decode_ms", 0) > 0
        for m in all_metrics
    )

    if has_phases:
        prec_labels = [m.precision for m in all_metrics]
        load_vals   = [m.business_output.get("load_ms", 0) for m in all_metrics]
        ttft_vals   = [m.business_output.get("ttft_ms", 0) for m in all_metrics]
        decode_vals = [m.business_output.get("decode_ms", m.latency_ms) for m in all_metrics]

        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(
            name="Load / Init", y=prec_labels, x=load_vals, orientation="h",
            marker_color="#6EA7D6",
        ))
        fig_wf.add_trace(go.Bar(
            name="Prefill (TTFT)", y=prec_labels, x=ttft_vals, orientation="h",
            marker_color="#E87722",
        ))
        fig_wf.add_trace(go.Bar(
            name="Decode", y=prec_labels, x=decode_vals, orientation="h",
            marker_color="#007DB8",
        ))
        fig_wf.update_layout(
            barmode="stack",
            height=max(220, len(all_metrics) * 70),
            xaxis_title="Time (ms)",
            showlegend=True,
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_wf, use_container_width=True)
    else:
        # LOB phase breakdown
        lob_metrics = [m for m in all_metrics if m.business_output.get("lob_gather_ms")]
        if lob_metrics:
            prec_labels = [m.precision for m in lob_metrics]
            fig_lob = go.Figure()
            fig_lob.add_trace(go.Bar(name="Gather", y=prec_labels, x=[m.business_output["lob_gather_ms"] for m in lob_metrics], orientation="h", marker_color="#E87722"))
            fig_lob.add_trace(go.Bar(name="Compute", y=prec_labels, x=[m.business_output["lob_compute_ms"] for m in lob_metrics], orientation="h", marker_color="#00A4E4"))
            fig_lob.add_trace(go.Bar(name="Scatter", y=prec_labels, x=[m.business_output["lob_scatter_ms"] for m in lob_metrics], orientation="h", marker_color="#007DB8"))
            fig_lob.update_layout(barmode="stack", height=max(200, len(lob_metrics) * 70), xaxis_title="Time (ms)", showlegend=True, legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_lob, use_container_width=True)
        else:
            st.info("No phase breakdown available for this scenario/workload.")

    st.divider()

    # -----------------------------------------------------------------------
    # 4. OPERATING CONDITION PANEL
    # -----------------------------------------------------------------------
    st.markdown("#### Operating Conditions")

    cond_rows = []
    for m in all_metrics:
        row = {"Precision": m.precision}
        oc = m.operational_condition
        row["Users"] = oc.get("num_users", 1)
        row["Batch"] = oc.get("batch_size", 1)
        row["Context"] = oc.get("context_length", "n/a")
        row["Seq len"] = oc.get("seq_len", "n/a")
        row["Fits VRAM"] = "✓" if oc.get("fits_in_memory", True) else "✗ SPILL"
        row["GPU util%"] = oc.get("gpu_util_pct", "n/a")
        row["Power (W)"] = oc.get("power_w", "n/a")
        row["Mem%"] = oc.get("mem_pct", "n/a")
        cond_rows.append(row)

    if cond_rows:
        st.dataframe(pd.DataFrame(cond_rows), use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------------------------------------------------
    # 5. TECHNICAL CHARTS (precision comparison — for technical audience)
    # -----------------------------------------------------------------------
    with st.expander("Technical precision charts", expanded=False):
        df_data = [m.to_dict() for m in all_metrics]
        df = pd.DataFrame(df_data)

        ch1, ch2 = st.columns(2)

        with ch1:
            st.subheader("Latency by Precision")
            fig_lat = go.Figure(go.Bar(
                x=df["precision"], y=df["latency_ms"],
                marker_color="#007DB8",
            ))
            fig_lat.update_layout(height=350, showlegend=False, yaxis_title="ms")
            st.plotly_chart(fig_lat, use_container_width=True)

        with ch2:
            st.subheader("Peak VRAM by Precision")
            fig_mem = go.Figure(go.Bar(
                x=df["precision"], y=df["peak_memory_mb"],
                marker_color="#003576",
            ))
            fig_mem.update_layout(height=350, showlegend=False, yaxis_title="MB")
            st.plotly_chart(fig_mem, use_container_width=True)

        if "tokens_per_sec" in df.columns and df["tokens_per_sec"].notna().any():
            tps_col = df["tokens_per_sec"].fillna(0)
            if tps_col.sum() > 0:
                st.subheader("Tokens/sec by Precision")
                fig_tps = go.Figure(go.Bar(
                    x=df["precision"], y=tps_col,
                    marker_color="#00A4E4",
                ))
                fig_tps.update_layout(height=300, showlegend=False, yaxis_title="tok/s")
                st.plotly_chart(fig_tps, use_container_width=True)

        fp32_rows = df[df["precision"] == "FP32"]
        if len(fp32_rows) > 0 and df["latency_ms"].gt(0).all():
            fp32_lat = fp32_rows.iloc[0]["latency_ms"]
            df["speedup"] = fp32_lat / df["latency_ms"]
            st.subheader("Speedup vs FP32")
            fig_sp = go.Figure(go.Bar(
                x=df["precision"], y=df["speedup"], marker_color="#007DB8"
            ))
            fig_sp.add_hline(y=1, line_dash="dash", line_color="#003576", annotation_text="FP32 baseline")
            fig_sp.update_layout(height=300, showlegend=False, yaxis_title="×")
            st.plotly_chart(fig_sp, use_container_width=True)

    # -----------------------------------------------------------------------
    # 6. GPU COST COMPARISON  (100× production scale)
    # -----------------------------------------------------------------------
    import math as _math

    st.divider()
    st.markdown("#### GPU Cost Comparison — 1,000× Production Scale")
    st.caption(
        "Models 1,000× the load of this benchmark run (users · throughput · concurrency). "
        "Throughput scaled by memory-bandwidth ratio. "
        "SXM GPUs aggregate via NVLink (no penalty). Non-SXM that cannot fit the model incur a 1.5× cost penalty. "
        "Scaling efficiency degrades ~8% per additional GPU in an NVLink aggregate. "
        "Economy of scale: 3% cost reduction per 10 nodes beyond 10 nodes. "
        "⚠ Estimates assume ideal NVLink topology — actual results may vary ±10% depending on cluster topology, workload communication pattern, and interconnect congestion."
    )

    # GPU catalogue — prices, specs, interconnect, SXM capability
    _GPU_COMPARE = {
        "RTX PRO 6000 BW":  {"price": 13_255, "mem_gb":  96, "bw_gbs":    960, "tdp_w":   300,
                              "color": "#6EA7D6", "sxm": False,
                              "ic_type": "PCIe 5.0",  "ic_bw_gbs":   128, "ic_premium": 0.00},
        "H100 SXM5 80GB":   {"price": 22_000, "mem_gb":  80, "bw_gbs":  3_350, "tdp_w":   700,
                              "color": "#007DB8", "sxm": True,
                              "ic_type": "NVLink 4.0", "ic_bw_gbs":   900, "ic_premium": 0.05},
        "H200 SXM 141GB":   {"price": 28_000, "mem_gb": 141, "bw_gbs":  4_800, "tdp_w":   700,
                              "color": "#005B8E", "sxm": True,
                              "ic_type": "NVLink 4.0", "ic_bw_gbs":   900, "ic_premium": 0.05},
        "B200 SXM 192GB":   {"price": 50_000, "mem_gb": 192, "bw_gbs":  8_000, "tdp_w": 1_000,
                              "color": "#003576", "sxm": True,
                              "ic_type": "NVLink 5.0", "ic_bw_gbs": 1_800, "ic_premium": 0.08},
        "GB200 NVL2 384GB": {"price": 75_000, "mem_gb": 384, "bw_gbs": 16_000, "tdp_w": 2_700,
                              "color": "#00A4E4", "sxm": True,
                              "ic_type": "NVLink 5.0", "ic_bw_gbs": 3_600, "ic_premium": 0.08},
        "GB300 NVL2 (est.)":{"price": 90_000, "mem_gb": 384, "bw_gbs": 18_000, "tdp_w": 2_800,
                              "color": "#E87722", "sxm": True,
                              "ic_type": "NVLink 5.0", "ic_bw_gbs": 3_600, "ic_premium": 0.10},
    }
    _GB10_BW_GBS   = 273          # real Dell Pro Max GB10 LPDDR5X memory BW (not NVLink-C2C 900)
    _SCALE_FACTOR  = 1000         # 1,000× production scale
    _OOM_PENALTY   = 1.5          # cost multiplier for non-SXM that can't fit model
    _AMORT_HOURS   = 3 * 365 * 24 # 3-year straight-line amortisation
    _POWER_RATE    = 0.12         # $/kWh

    _model_gb      = best.peak_memory_mb / 1024
    _gb10_tps      = best.business_output.get("tokens_per_sec", best.tokens_per_sec) or 0
    _gb10_ips      = best.business_output.get("images_per_sec", 0) or 0
    _gb10_bws      = best.business_output.get("bandwidth_gbs", 0) or 0
    _primary_tput  = _gb10_tps or _gb10_ips or _gb10_bws
    _tput_label    = "Tok/s" if _gb10_tps > 0 else ("Img/s" if _gb10_ips > 0 else "GB/s")
    _target_tput   = _primary_tput * _SCALE_FACTOR

    gpu_rows, bar_cpm_labels, bar_cpm_vals, bar_cpm_colors = [], [], [], []
    bar_capex_labels, bar_capex_vals, bar_capex_colors      = [], [], []

    for _gname, _g in _GPU_COMPARE.items():
        _fits_single = _model_gb <= _g["mem_gb"]
        _ic_prem     = _g.get("ic_premium", 0.0)

        if _fits_single:
            _gpus_per_node = 1
            _node_bw       = _g["bw_gbs"]
            _cost_mult     = 1.0 + _ic_prem
            _fit_status    = "✓ Fits"
            _scale_eff_pct = 100.0
        elif _g["sxm"]:
            _gpus_per_node = _math.ceil(_model_gb / _g["mem_gb"])
            _node_bw       = _g["bw_gbs"] * _gpus_per_node
            _cost_mult     = 1.0 + _ic_prem
            _fit_status    = f"NVLink ×{_gpus_per_node}"
            # Scaling efficiency: 100% at 1 GPU, -8% per additional GPU, floor 70%
            _scale_eff_pct = max(70.0, 100.0 - (_gpus_per_node - 1) * 8.0)
        else:
            _gpus_per_node = 1
            _node_bw       = _g["bw_gbs"]
            _cost_mult     = _OOM_PENALTY + _ic_prem
            _fit_status    = f"✗ OOM ×{_OOM_PENALTY:.1f}"
            _scale_eff_pct = 50.0  # heavy penalty for single-GPU OOM situation

        # Effective node throughput with scaling efficiency applied
        _node_tput = (_primary_tput * (_node_bw / _GB10_BW_GBS) * (_scale_eff_pct / 100.0)
                      if _primary_tput > 0 else 0)

        if _node_tput > 0 and _target_tput > 0:
            _nodes_needed = max(1, _math.ceil(_target_tput / _node_tput))
        else:
            _nodes_needed = None

        _total_gpus = _nodes_needed * _gpus_per_node if _nodes_needed else None

        # Economy of scale: 3% reduction per 10 nodes beyond 10 nodes
        if _nodes_needed:
            _eco_factor = max(0.75, 1.0 - 0.03 * max(0, (_nodes_needed - 10) // 10))
        else:
            _eco_factor = 1.0

        _effective_price = _g["price"] * _cost_mult * _eco_factor
        _total_hw   = _total_gpus * _effective_price if _total_gpus else None
        _total_tdp  = _total_gpus * _g["tdp_w"] if _total_gpus else None

        if _total_hw and _total_tdp and _target_tput > 0:
            _hw_per_hr    = _total_hw / _AMORT_HOURS
            _pwr_per_hr   = _total_tdp * _POWER_RATE / 1000
            _total_per_hr = _hw_per_hr + _pwr_per_hr
            _cpm = (_total_per_hr / (_target_tput * 3600)) * 1_000_000
        else:
            _cpm = None

        gpu_rows.append({
            "GPU":                    _gname,
            "Mem":                    f"{_g['mem_gb']} GB",
            "Interconnect":           _g.get("ic_type", "—"),
            "IC BW":                  f"{_g.get('ic_bw_gbs', 0):,} GB/s",
            "Model Fit":              _fit_status,
            "Scale Eff.":             f"{_scale_eff_pct:.0f}%",
            "Node Size":              f"×{_gpus_per_node}" if _gpus_per_node > 1 else "1 GPU",
            f"Nodes (1000×)":         f"{_nodes_needed:,}" if _nodes_needed else "—",
            "Total GPUs":             f"{_total_gpus:,}" if _total_gpus else "—",
            "Total CapEx":            f"${_total_hw:,.0f}" if _total_hw else "—",
            f"$/M{_tput_label}":      f"${_cpm:.4f}" if _cpm else "—",
        })

        if _cpm is not None:
            bar_cpm_labels.append(_gname)
            bar_cpm_vals.append(_cpm)
            bar_cpm_colors.append(_g["color"])
        if _total_hw is not None:
            bar_capex_labels.append(_gname)
            bar_capex_vals.append(_total_hw)
            bar_capex_colors.append(_g["color"])

    st.dataframe(pd.DataFrame(gpu_rows), use_container_width=True, hide_index=True)

    if bar_cpm_vals and len(bar_cpm_vals) > 1:
        _cc1, _cc2 = st.columns(2)

        with _cc1:
            st.markdown(f"**$/M{_tput_label} at 1,000× Load**")
            _fig_cpm = go.Figure(go.Bar(
                x=bar_cpm_labels, y=bar_cpm_vals,
                marker_color=bar_cpm_colors,
                text=[f"${v:.4f}" for v in bar_cpm_vals],
                textposition="outside", textfont=dict(size=10),
            ))
            _fig_cpm.update_layout(
                height=360, yaxis_title=f"$ per Million {_tput_label}",
                showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(_fig_cpm, use_container_width=True)

        with _cc2:
            st.markdown("**Total CapEx for 1,000× Workload**")
            _fig_capex = go.Figure(go.Bar(
                x=bar_capex_labels, y=bar_capex_vals,
                marker_color=bar_capex_colors,
                text=[f"${v/1e6:.2f}M" if v >= 1e6 else f"${v:,.0f}" for v in bar_capex_vals],
                textposition="outside", textfont=dict(size=10),
            ))
            _fig_capex.update_layout(
                height=360, yaxis_title="Total Hardware Cost ($)",
                showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=40, b=10),
                yaxis=dict(tickformat="$,.0f"),
            )
            st.plotly_chart(_fig_capex, use_container_width=True)

    st.caption(
        f"Load basis: {_primary_tput:,.0f} {_tput_label} on Dell Pro Max GB10 → {_target_tput:,.0f} {_tput_label} target (1,000×). "
        "3-year straight-line CapEx amortization at $0.12/kWh power. "
        "Scaling efficiency: −8%/GPU in NVLink aggregate (floor 70%); 50% for OOM non-SXM configs. "
        "Economy of scale: −3% per 10 nodes beyond 10. "
        "⚠ ±10% variance expected depending on cluster topology, workload communication pattern, and interconnect congestion."
    )

    # -----------------------------------------------------------------------
    # 7. MODEL SCALE RECOMMENDATION
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("#### Model & Scale Recommendation")

    def _scale_advice(scenario: str, model_name: str, best_m, all_m: list) -> dict:
        """Return scale recommendation dict for this scenario + benchmark result."""
        tps   = best_m.business_output.get("tokens_per_sec", best_m.tokens_per_sec) or 0
        ips   = best_m.business_output.get("images_per_sec", 0) or 0
        mem   = best_m.peak_memory_mb / 1024
        prec  = best_m.precision
        model = model_name.split("/")[-1]

        if scenario in ("Quick Inference", "Long Context"):
            return {
                "best_for_scale": "Qwen/Qwen2.5-7B-Instruct or mistralai/Mistral-7B-v0.1",
                "why": (
                    "7B-class models deliver the best TTFT-to-cost ratio at any scale. "
                    "At low latency budgets (<200 ms TTFT) they out-compete larger models. "
                    "For quality-critical deployments step up to Phi-4 (14B-class quality at 14B memory)."
                ),
                "scale_gpu": "H100 SXM5 (up to 8 sessions/GPU) → H200 (11 sessions/GPU at 141 GB)",
                "cost_gpu": "RTX PRO 6000 Blackwell cluster — best $/MTok below 32B model size",
                "prec_advice": "BF16 for quality; NVFP4 on Blackwell for 4× throughput at near-FP16 quality",
                "gb10_equiv": f"~{max(1, int(30_000/3_000))} Dell Pro Max GB10 units ≈ 1 H100 on raw BW; Dell Pro Max GB10 wins on $/MTok",
            }

        elif scenario == "Batch Throughput":
            return {
                "best_for_scale": "mistralai/Mixtral-8x7B-Instruct-v0.1 or Qwen/Qwen2.5-14B-Instruct",
                "why": (
                    "MoE (Mixtral) routes only 2 of 8 experts per token, giving near-7B latency "
                    "with 70B-class quality. At large batch the Dell Pro Max GB10's 128 GB unified memory "
                    "lets you pack bigger batches than an H100 (80 GB), but H200 (141 GB) "
                    "wins at batch≥64."
                ),
                "scale_gpu": "H200 SXM (best memory per dollar for large-batch) → B200 for FP4 acceleration",
                "cost_gpu": "GB200 NVL2 at scale — 384 GB fits 72B models for maximum throughput",
                "prec_advice": "INT8 or NVFP4 to double/quadruple batch capacity within the same VRAM budget",
                "gb10_equiv": f"This run: {tps:.0f} tok/s · H200 est. {tps*(4800/273):.0f} tok/s (~18× Dell Pro Max GB10 mem BW)",
            }

        elif scenario == "Batch Limit Sweep":
            return {
                "best_for_scale": "Largest model that fits in 80% of target GPU VRAM at your working precision",
                "why": (
                    "OOM boundary shifts proportionally with VRAM. If the Dell Pro Max GB10 OOM'd at batch N "
                    "with 128 GB, an H200 (141 GB) extends it ~1.1×; a B200 (192 GB) ~1.5×; "
                    "GB200 NVL2 (384 GB) ~3×."
                ),
                "scale_gpu": "B200 SXM (192 GB) for 1.5× OOM ceiling over Dell Pro Max GB10 at same model",
                "cost_gpu": "Dell Pro Max GB10 cluster for $/MTok; single B200 for peak throughput per U of rack",
                "prec_advice": "NVFP4 on Blackwell GPUs effectively halves model footprint, doubling the OOM ceiling",
                "gb10_equiv": f"Model used {mem:.1f} GB of 128 GB ({mem/1.28:.0f}% fill)",
            }

        elif scenario == "Multi-User Concurrency":
            return {
                "best_for_scale": "TinyLlama/TinyLlama-1.1B-Chat-v1.0 (max sessions) or Qwen2.5-7B (quality+scale)",
                "why": (
                    "Session count scales inversely with model size. 1B fits ~100 sessions in 128 GB; "
                    "7B fits ~14 sessions; 14B fits ~7 sessions. "
                    "Dell Pro Max GB10's 128 GB unified memory outperforms H100 (80 GB) for concurrent session count."
                ),
                "scale_gpu": "GB200 NVL2 (384 GB) → 3× the concurrent sessions of Dell Pro Max GB10 per unit",
                "cost_gpu": "Dell Pro Max GB10 cluster — $3K/unit, scale out horizontally for enterprise session counts",
                "prec_advice": "NVFP4 or INT8 to double effective session capacity within same VRAM",
                "gb10_equiv": f"128 GB unified → can serve ~{int(128 / max(0.1, mem))} sessions at {mem:.1f} GB/session",
            }

        elif scenario == "Memory Pressure":
            return {
                "best_for_scale": "Use NVFP4 quantized models (nvidia/Qwen3-8B-NVFP4) for maximum memory efficiency",
                "why": (
                    "FP4 reduces model footprint 4× vs FP16. On Blackwell hardware (Dell Pro Max GB10, B200, GB200) "
                    "NVFP4 is hardware-accelerated with near-FP16 accuracy. "
                    "For memory-bandwidth stress testing, B200/GB200 HBM3e delivers 8–16 TB/s vs Dell Pro Max GB10's 0.27 TB/s LPDDR5X."
                ),
                "scale_gpu": "B200 SXM — 8 TB/s HBM3e BW, ~29× higher sustained bandwidth than Dell Pro Max GB10",
                "cost_gpu": "Dell Pro Max GB10 is unmatched at ~$4K; next step is RTX PRO 6000 at ~$13K with 960 GB/s GDDR7",
                "prec_advice": "NVFP4 on any Blackwell GPU; BF16 on H200 for HBM3e bandwidth comparison",
                "gb10_equiv": f"Dell Pro Max GB10 at {mem:.1f} GB used · BW scaling: H200 = {tps*(4800/273):.0f} vs {tps:.0f} Dell Pro Max GB10",
            }

        elif scenario == "Vision Throughput":
            return {
                "best_for_scale": "openai/clip-vit-large-patch14 (quality) or facebook/dino-vits16 (speed/cost)",
                "why": (
                    "ViT/CLIP throughput scales linearly with memory bandwidth up to compute-saturation. "
                    "At batch≥64 the workload becomes compute-bound (TFlops), where B200 and GB200 "
                    "pull far ahead. For edge/workstation scale, RTX PRO 6000 at ~$13K "
                    "gives 960 GB/s GDDR7 — ~3.5× Dell Pro Max GB10's memory bandwidth — at a fraction of H100 cost."
                ),
                "scale_gpu": "H100 SXM5 → H200 for large-batch image pipelines (>10K img/s)",
                "cost_gpu": "RTX PRO 6000 Blackwell — best img/s per dollar for sub-1K batch workloads",
                "prec_advice": "FP16/BF16 for quality; INT8 for 2× throughput at batch≥32",
                "gb10_equiv": f"Dell Pro Max GB10: {ips:.1f} img/s · H100 est. {ips*(3350/273):.0f} img/s at ~12× BW",
            }

        else:  # HPC / Quant Analysis
            return {
                "best_for_scale": "No specific model — raw GPU compute; scale by TFlops not model size",
                "why": (
                    "HPC workloads (MatMul, LOB, bandwidth) are compute- or bandwidth-bound by GPU specs. "
                    "B200 delivers 2,250 TFLOPS BF16 (2.3× H100); GB200 NVL2 delivers 4,500 TFLOPS "
                    "(4.6× H100). For pure bandwidth stress, H200 HBM3e at 4.8 TB/s is the sweet spot "
                    "before stepping to NVL-scale hardware."
                ),
                "scale_gpu": "GB200 NVL2 for peak TFLOPS; H200 for peak memory bandwidth per dollar",
                "cost_gpu": "Dell Pro Max GB10 unbeatable for edge HPC/financial compute at $3K — 500 TFLOPS BF16",
                "prec_advice": "FP64 for scientific accuracy; BF16/TF32 for ML HPC; FP4 for AI-accelerated quant",
                "gb10_equiv": "Dell Pro Max GB10 ≈ 500 TFLOPS BF16 · B200 = 4.5× · GB200 NVL2 = 9×",
            }

    _advice = _scale_advice(scenario, rdata["model"], best, all_metrics)

    ac1, ac2 = st.columns([3, 2])
    with ac1:
        st.markdown(f"""
<div class="biz-card">
<h4>Best Model for This Scenario at Scale</h4>
<b>{_advice['best_for_scale']}</b>
<p style="margin:10px 0 0 0; font-size:0.9em; color:#444;">{_advice['why']}</p>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="biz-card">
<h4>Precision Recommendation</h4>
<p style="font-size:0.9em; color:#444;">{_advice['prec_advice']}</p>
</div>
""", unsafe_allow_html=True)

    with ac2:
        st.markdown(f"""
<div class="biz-card">
<h4>Best GPU for Throughput</h4>
<p style="font-size:0.9em; color:#444;">{_advice['scale_gpu']}</p>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="biz-card">
<h4>Best GPU for Cost Efficiency</h4>
<p style="font-size:0.9em; color:#444;">{_advice['cost_gpu']}</p>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="biz-card">
<h4>Dell Pro Max GB10 Scale Context</h4>
<p style="font-size:0.9em; color:#444;">{_advice['gb10_equiv']}</p>
</div>
""", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 8. CSV EXPORT — all 4 dimensions
    # -----------------------------------------------------------------------
    st.divider()
    df_all = pd.DataFrame([m.to_dict() for m in all_metrics])
    csv = df_all.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Results (CSV — all 4 dimensions)",
        data=csv,
        file_name=f"gb10_{rdata['scenario'].replace(' ', '_')}_{rdata['model'].replace('/', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# TAB 4 — TCO Analysis
# ---------------------------------------------------------------------------
with tab_tco:
    st.subheader("Total Cost of Ownership — Dell HW Lineup")
    st.caption(
        "Compare CapEx + 3-year OpEx across Dell PowerEdge XE systems. "
        "Projections use measured Dell Pro Max GB10 benchmark throughput (when available) scaled "
        "by memory-bandwidth ratio. Prices are approximate 2025-2026 list prices."
    )

    # Workload profile — drives how the Best/Better/Good rating is scored.
    #   LLM Inference          → per-user decode tok/s blended 50/50 with $/MTok cost.
    #   FinTech / Bandwidth-bound → memory bandwidth + VRAM capacity (cost-independent);
    #     for quant/HFT (LOB scatter/gather, Black-Scholes, risk) which are memory-bound.
    _PROFILE_LABELS = {
        "LLM Inference (tok/s + $/MTok)": "llm",
        "FinTech / Bandwidth-bound (memory speed + capacity)": "fintech",
    }
    tco_profile_label = st.radio(
        "Rating profile",
        options=list(_PROFILE_LABELS.keys()),
        horizontal=True,
        help="LLM Inference rates systems on per-user token speed and cost. "
             "FinTech / Bandwidth-bound rates them on raw per-GPU memory bandwidth and "
             "VRAM capacity — the ceiling for memory-bound quant workloads (limit order "
             "book, Black-Scholes, risk reductions), independent of token cost.",
    )
    tco_profile = _PROFILE_LABELS[tco_profile_label]
    if tco_profile == "fintech":
        st.info(
            "📈 **FinTech / Bandwidth-bound profile** — driven by **Monte-Carlo test data** "
            "(a working set of paths) and a **target paths/sec**, not an LLM model. Each system "
            "is sized to `max(fit the working set, meet the demanded bandwidth)`, then ranked by "
            "**fewest GPUs** (Best/Better/Good rating = per-GPU memory speed 70% + VRAM capacity 30%). "
            "High-HBM parts need far fewer GPUs than Dell Pro Max GB10's 273 GB/s LPDDR5X — for bandwidth-bound "
            "quant, Dell Pro Max GB10 is honestly the wrong tool, and this surfaces that."
        )

    # -----------------------------------------------------------------------
    # Pull Dell Pro Max GB10 measured TPS from session state (if benchmark has been run)
    # -----------------------------------------------------------------------
    _results = st.session_state.get("results", {})
    _gb10_tps_measured: float | None = None
    _gb10_tps_model: str | None = None
    if _results:
        # Prefer the highest token/sec across measured precisions (LLM scenarios).
        # BenchmarkMetrics stores decode throughput on `.tokens_per_sec`.
        for _m in _results.get("metrics", []):
            _tps = 0.0
            if isinstance(_m, dict):
                _tps = float(_m.get("tokens_per_sec") or 0)
            elif hasattr(_m, "tokens_per_sec"):
                _tps = float(getattr(_m, "tokens_per_sec", 0) or 0)
            if _tps > (_gb10_tps_measured or 0):
                _gb10_tps_measured = _tps
                _gb10_tps_model = _results.get("model")

    # -----------------------------------------------------------------------
    # TCO Configuration Controls
    # -----------------------------------------------------------------------
    tco_c1, tco_c2, tco_c3 = st.columns([1, 1, 1])

    # FinTech profile is driven by Monte-Carlo "test data" (a working set of paths)
    # and a paths/sec sizing target — not by an LLM model / precision / tokens.
    is_fintech = tco_profile == "fintech"
    # Defaults so later (LLM-only) references never NameError under the fintech branch.
    tco_model = tco_precision = None
    tco_users = tco_output_toks = tco_ctx = tco_tps_override = 0
    tco_employees = 0
    wf = None
    mc_resident = mc_bytes = mc_steps = mc_target = 0

    with tco_c1:
        if is_fintech:
            st.markdown("**Monte-Carlo Test Data**")
            mc_preset = st.selectbox(
                "Path model (per-path state)",
                options=list(MC_PATH_PRESETS.keys()),
                index=3,
                help="Sets bytes of path-state swept each timestep — the 'test data' "
                     "complexity. Heston/basket/VaR touch more memory per path.",
            )
            mc_bytes = MC_PATH_PRESETS[mc_preset]["bytes"]
            st.caption(f"{mc_bytes} B/path · {MC_PATH_PRESETS[mc_preset]['desc']}")
            mc_resident = st.number_input(
                "Resident paths (working set)",
                min_value=1e5, max_value=1e10, value=5e7, step=1e6, format="%.0f",
                help="Path-state held in VRAM at once. × bytes/path = working set that "
                     "must fit. Embarrassingly parallel — shards across GPUs/nodes.",
            )
            mc_steps = st.number_input(
                "Timesteps / path",
                min_value=1, max_value=10_000, value=252, step=1,
                help="Sweeps over the path state to complete one path (e.g. 252 "
                     "trading days). Raises memory traffic, not the working set.",
            )
            mc_target = st.number_input(
                "Target paths / sec  (sizing need)",
                min_value=1e6, max_value=1e12, value=1e9, step=1e8, format="%.0f",
                help="Throughput to provision for. Demanded bandwidth = "
                     "paths/sec × bytes/path × timesteps → drives GPUs needed.",
            )
        else:
            st.markdown("**Workload Parameters**")
            tco_model = st.selectbox(
                "Target Model",
                options=list(MODEL_CATALOG.keys()),
                index=list(MODEL_CATALOG.keys()).index("Llama-3.3-70B")
                if "Llama-3.3-70B" in MODEL_CATALOG else 0,
                help="Includes models too large for a single Dell Pro Max GB10 — system will auto-scale nodes.",
            )
            _tco_prec_opts = supported_precisions(tco_model)
            _tco_native = native_precision(tco_model)
            tco_precision = st.selectbox(
                "Inference Precision",
                options=_tco_prec_opts,
                index=_tco_prec_opts.index(_tco_native) if _tco_native in _tco_prec_opts else 0,
                help="Limited to the precisions this checkpoint actually runs at "
                     "(e.g. a pre-quantized NVFP4 or FP8-native model won't run FP16/BF16).",
            )
            tco_employees = st.number_input(
                "Total Employees",
                min_value=10, max_value=100_000, value=500, step=10,
                help="Whole workforce this deployment serves. Sizing is derived from the "
                     "usage-tier mix below — not everyone is active at once, and a few "
                     "power users drive most of the tokens.",
            )
            _wf_c1, _wf_c2 = st.columns(2)
            with _wf_c1:
                _wf_power_pct = st.slider(
                    "Power users (% of staff)", 1, 20,
                    int(WORKFORCE_DEFAULTS["power_pct"]),
                    help="Heavy, near-continuous users (engineers, analysts). Few, but "
                         "they generate the majority of tokens.",
                )
            with _wf_c2:
                _wf_minimal_pct = st.slider(
                    "Minimal users (% of staff)", 10, 50,
                    int(WORKFORCE_DEFAULTS["minimal_pct"]),
                    help="Occasional users. General users = the remainder.",
                )
            with st.expander("Per-tier concurrency (active at peak)"):
                _wf_conc_power = st.slider(
                    "Power concurrency %", 10, 100,
                    int(WORKFORCE_DEFAULTS["conc_power"]), step=5)
                _wf_conc_general = st.slider(
                    "General concurrency % (baseline)", 1, 60,
                    int(WORKFORCE_DEFAULTS["conc_general"]), step=1)
                _wf_conc_minimal = st.slider(
                    "Minimal concurrency %", 0, 30,
                    int(WORKFORCE_DEFAULTS["conc_minimal"]), step=1)

            wf = workforce_demand(
                tco_employees,
                power_pct=_wf_power_pct, minimal_pct=_wf_minimal_pct,
                conc_power=_wf_conc_power, conc_general=_wf_conc_general,
                conc_minimal=_wf_conc_minimal,
            )
            # Effective concurrent sessions drive the rest of the TCO math unchanged.
            tco_users = wf.effective_sessions
            if wf.warning:
                st.warning(wf.warning)
            st.caption(
                f"General users: **{wf.general_pct:.0f}%** · "
                f"{tco_employees:,} employees → **~{wf.effective_sessions:,}** effective "
                f"concurrent sessions ({wf.effective_concurrency:.0%} of staff)"
            )
            tco_output_toks = st.slider(
                "Output Tokens / Request",
                min_value=256, max_value=16_384, value=16_384, step=256,
                help="Average tokens generated per user request. Longer outputs grow "
                     "the KV cache, consuming VRAM and reducing sessions per node.",
            )
            tco_ctx = st.select_slider(
                "Context Length",
                options=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                         65536, 131072, 262144, 524288],
                value=512,
                format_func=lambda v: f"{v // 1024}K" if v >= 1024 else str(v),
            )

    with tco_c2:
        st.markdown("**Financial Parameters**")
        tco_amort = st.slider(
            "Amortization Period (years)",
            min_value=1, max_value=5, value=3,
        )
        tco_power = st.slider(
            "Power Cost ($/kWh)",
            min_value=0.05, max_value=0.30, value=0.12, step=0.01,
            format="$%.2f",
        )
        tco_infra_pct = st.slider(
            "Infra Overhead (networking, storage, cooling %)",
            min_value=0, max_value=30, value=15, step=1,
        ) / 100.0

        # Dell Pro Max GB10 measured TPS override (LLM profile only — FinTech sizes on bandwidth).
        if not is_fintech:
            st.markdown("**Throughput Baseline**")
            if _gb10_tps_measured:
                _src = f" ({_gb10_tps_model.split('/')[-1]})" if _gb10_tps_model else ""
                st.success(f"Using measured Dell Pro Max GB10 TPS: **{_gb10_tps_measured:.0f} tok/s**{_src}")
                default_tps = _gb10_tps_measured
            else:
                st.info("No benchmark run yet — using estimated Dell Pro Max GB10 throughput.")
                default_tps = None

            # Measured TPS can legitimately be <10 tok/s for very large models, so
            # keep min low and clamp the default into [min,max] — a measured value
            # outside the range must not crash the widget (StreamlitValueBelowMinError).
            _tps_min, _tps_max = 1.0, 50000.0
            _tps_default = float(default_tps) if default_tps else 1200.0
            _tps_default = min(_tps_max, max(_tps_min, _tps_default))
            tco_tps_override = st.number_input(
                "Dell Pro Max GB10 Baseline TPS (tok/s) — override",
                min_value=_tps_min, max_value=_tps_max,
                value=_tps_default,
                step=50.0,
                help="Tokens/sec measured on Dell Pro Max GB10 for this model+precision. Used to scale projections.",
            )

    with tco_c3:
        st.markdown("**Systems to Compare**")
        # Scope gates the catalog (~76 systems): defaulting to all of them would mean 76 TCO
        # runs, a 76-row table and unreadable 76-bar charts on first paint. The shortlist is
        # one platform per row, each on a different GPU; everything else is a scope away.
        _SCOPE_LABELS = {
            "Default shortlist": "default",
            "All 17G":           "17G",
            "All 16G":           "16G",
            "Everything":        "all",
        }
        tco_scope = _SCOPE_LABELS[st.radio(
            "Catalog scope",
            options=list(_SCOPE_LABELS.keys()),
            horizontal=True,
            key="tco_scope",
            help="Default shows a curated shortlist. Widen to reach the full Dell matrix.",
        )]
        if tco_scope == "default":
            _scope_names = [s for s in DEFAULT_SYSTEMS if s in DELL_SYSTEMS]
        elif tco_scope == "all":
            _scope_names = list(DELL_SYSTEMS.keys())
        else:
            # Always keep the Dell Pro Max GB10 baseline visible — it is what every projection scales from.
            _scope_names = ["Dell Pro Max GB10"] + [
                n for n, s in DELL_SYSTEMS.items()
                if s.get("generation") == tco_scope and n != "Dell Pro Max GB10"
            ]
        # options = the WHOLE catalog, always; scope only chooses what starts selected. If options
        # were limited to the scope, everything in it would already be selected and the dropdown
        # would open EMPTY with nothing left to add — the point is that any of the other systems
        # stays one dropdown away.
        # Explicit key (varying with scope) so switching scope re-applies that scope's default
        # rather than Streamlit's positional auto-key holding the previous selection.
        tco_systems = st.multiselect(
            "Select Dell Systems",
            options=list(DELL_SYSTEMS.keys()),
            default=_scope_names,
            key=f"tco_systems_{tco_scope}",
            help="Scope sets the starting selection; add any other system from this list.",
        )

    if not tco_systems:
        st.warning("Select at least one system to compare.")
        st.stop()

    # Precision compatibility — if the chosen precision can't run on this model's
    # checkpoint (e.g. pre-quantized NVFP4 at FP16), show Not Compatible and do
    # not compute a TCO for it. Quantization precisions stay valid. (LLM only —
    # FinTech has no model/precision.)
    if not is_fintech:
        _tco_ok, _tco_why = precision_compatible(tco_model, tco_precision)
        if not _tco_ok:
            st.warning(f"⛔ **Not Compatible** — {tco_model} at {tco_precision}: {_tco_why}. "
                       "No TCO computed; choose FP4/NVFP4 for this model.")
            st.stop()

    # -----------------------------------------------------------------------
    # Platform price reference — list price of each platform in the comparison
    # -----------------------------------------------------------------------
    with st.expander("💲 Platform price reference (per-node / per-rack list price)", expanded=False):
        _price_rows = []
        for _sn in tco_systems:
            _si = DELL_SYSTEMS[_sn]
            _gpus = _si["gpus_per_node"]
            _price_rows.append({
                "Platform":       _sn,
                "GPUs / Unit":    _gpus,
                "GPU":            _si["gpu_model"],
                "Link":           _si.get("link_label", "—"),
                "Network":        _si.get("net", "—"),
                "Unit List $":    format_usd(_si["system_price"]),
                "$ / GPU":        format_usd(_si["system_price"] / max(_gpus, 1)),
            })
        st.dataframe(pd.DataFrame(_price_rows), use_container_width=True, hide_index=True)
        st.caption("Approximate 2025-2026 list prices. CapEx in the comparison adds "
                   f"{tco_infra_pct:.0%} infra overhead (networking, storage, cooling) on top.")

    st.divider()

    # -----------------------------------------------------------------------
    # Workload summary
    # -----------------------------------------------------------------------
    if is_fintech:
        _working_gb   = mc_resident * mc_bytes / 1e9
        _demanded_tbs = mc_target * mc_bytes * mc_steps / 1e12
        ma1, ma2, ma3, ma4 = st.columns(4)
        ma1.metric("Resident Paths", f"{mc_resident/1e6:,.0f}M")
        ma2.metric("Working Set", f"{_working_gb:,.1f} GB",
                   help=f"{mc_resident:,.0f} paths × {mc_bytes} B/path — must fit in VRAM")
        ma3.metric("Demanded BW", f"{_demanded_tbs:,.2f} TB/s",
                   help=f"{mc_target:,.0f} paths/s × {mc_bytes} B × {mc_steps} steps")
        ma4.metric("Fits on Dell Pro Max GB10 (128 GB)?", "Yes" if _working_gb <= 128 else "No")
    else:
        _eff_ctx = tco_ctx + tco_output_toks   # KV cache holds context + generated output
        m_gb = model_memory_gb(tco_model, tco_precision, batch_size=1, context_len=_eff_ctx)
        info = MODEL_CATALOG.get(tco_model, {})
        ma1, ma2, ma3, ma4 = st.columns(4)
        ma1.metric("Model Params", f"{info.get('params_b', 0):.1f}B")
        ma2.metric("VRAM / Session", f"{m_gb:.1f} GB",
                   help=f"Weights + KV cache for {_eff_ctx:,} tokens "
                        f"({tco_ctx:,} ctx + {tco_output_toks:,} output)")
        ma3.metric("Precision", tco_precision)
        ma4.metric("Fits on Dell Pro Max GB10 (128 GB)?", "Yes" if m_gb <= 128 else "No")

    # -----------------------------------------------------------------------
    # Workforce demand — total employees → effective concurrent sessions
    # (LLM profile only; FinTech sizes on Monte-Carlo bandwidth, not seats)
    # -----------------------------------------------------------------------
    if not is_fintech and wf is not None:
        st.markdown("### Workforce Demand")
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("Total Employees", f"{wf.total_employees:,}")
        wc2.metric("Effective Concurrent Sessions", f"{wf.effective_sessions:,}",
                   help="Σ (headcount × per-tier concurrency) — what the hardware is sized for.")
        wc3.metric("Effective Concurrency", f"{wf.effective_concurrency:.0%}",
                   help="Effective sessions ÷ total employees. Rises above the 20% baseline "
                        "when the workforce is power-heavy.")
        wc4.metric("Power-user Token Share", f"{wf.token_share['power']:.0f}%",
                   help="Share of all tokens consumed by power users — a thin sliver of staff "
                        "drives most of the load.")

        _WF_LABELS = {"power": "Power", "general": "General", "minimal": "Minimal"}
        _WF_COLORS = {"power": "#007DB8", "general": "#00A4E4", "minimal": "#E87722"}
        _rows = ["Tokens", "Workforce"]   # bottom-to-top: workforce on top
        fig_wf_mix = go.Figure()
        for _t in ("power", "general", "minimal"):
            fig_wf_mix.add_trace(go.Bar(
                name=_WF_LABELS[_t], orientation="h", y=_rows,
                x=[wf.token_share[_t], wf.headcount_pct[_t]],
                marker_color=_WF_COLORS[_t],
                customdata=[[wf.active_sessions[_t]], [wf.headcount[_t]]],
                hovertemplate=("%{y} — " + _WF_LABELS[_t]
                               + ": %{x:.0f}%<br>count: %{customdata[0]:,}<extra></extra>"),
                text=[f"{wf.token_share[_t]:.0f}%", f"{wf.headcount_pct[_t]:.0f}%"],
                textposition="inside", insidetextanchor="middle",
            ))
        fig_wf_mix.update_layout(
            barmode="stack", height=200,
            xaxis=dict(title="Share (%)", range=[0, 100], ticksuffix="%"),
            legend=dict(orientation="h", y=1.25),
            margin=dict(l=10, r=10, t=30, b=10),
            uniformtext=dict(mode="hide", minsize=10),
        )
        st.plotly_chart(fig_wf_mix, use_container_width=True)
        st.caption(
            f"**Workforce** = headcount split ({wf.headcount['power']:,} power · "
            f"{wf.headcount['general']:,} general · {wf.headcount['minimal']:,} minimal). "
            f"**Tokens** = share of consumption — {wf.token_share['power']:.0f}% from the "
            f"{wf.headcount_pct['power']:.0f}% who are power users. Concurrent is no longer the "
            f"only factor: token intensity per tier reshapes real load."
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Run TCO for selected systems
    # -----------------------------------------------------------------------
    _results_by_sys = {}
    for sname in tco_systems:
        try:
            if is_fintech:
                _results_by_sys[sname] = calculate_tco_montecarlo(
                    system_name=sname,
                    resident_paths=mc_resident,
                    bytes_per_path=mc_bytes,
                    timesteps=int(mc_steps),
                    target_paths_per_sec=mc_target,
                    amort_years=tco_amort,
                    power_rate=tco_power,
                    add_infra_pct=tco_infra_pct,
                )
            else:
                _results_by_sys[sname] = calculate_tco(
                    system_name=sname,
                    model_name=tco_model,
                    precision=tco_precision,
                    num_users=tco_users,
                    gb10_tps=tco_tps_override,
                    output_toks=tco_output_toks,
                    context_len=tco_ctx,
                    amort_years=tco_amort,
                    power_rate=tco_power,
                    add_infra_pct=tco_infra_pct,
                )
        except Exception as exc:
            _results_by_sys[sname] = exc

    # Combined TCO + performance rating is computed across the whole peer group.
    _ok_results = [r for r in _results_by_sys.values() if not isinstance(r, Exception)]
    assign_ratings(_ok_results, profile=tco_profile)

    tco_rows = []
    for sname in tco_systems:
        r = _results_by_sys[sname]
        sys_info = DELL_SYSTEMS[sname]
        if isinstance(r, Exception):
            if is_fintech:
                tco_rows.append({
                    "_result": None, "System": sname, "GPU": sys_info["gpu_model"],
                    "BW/GPU": "—", "Working Set": "—", "Paths/s": "—",
                    "GPUs Needed": "—", "Nodes": "—", "Unit $": format_usd(sys_info["system_price"]),
                    "CapEx": "—", f"{tco_amort}yr TCO": "—", "$/B-paths": "—",
                    "Rating": f"Error: {r}",
                })
            else:
                tco_rows.append({
                    "_result": None, "System": sname, "GPU": sys_info["gpu_model"],
                    "Link": sys_info.get("link_label", "—"), "BW/GPU": "—",
                    "Decode Tok/s/User": "—", "Per-Copy": "—", "Nodes": "—",
                    "GPUs": "—", "Unit $": format_usd(sys_info["system_price"]),
                    "CapEx": "—", f"{tco_amort}yr TCO": "—", "$/MTok": "—",
                    "Rating": f"Error: {r}",
                })
            continue
        if is_fintech:
            tco_rows.append({
                "_result":            r,
                "System":             sname,
                "GPU":                sys_info["gpu_model"],
                "BW/GPU":             f"{r.mem_bw_tbs:.2f} TB/s",
                "Working Set":        f"{r.working_set_gb:,.1f} GB",
                "Paths/s":            f"{r.paths_per_sec/1e9:,.2f}B",
                "GPUs Needed":        f"{r.gpus_total:,}",
                "Nodes":              f"{r.num_nodes:,}",
                "Unit $":             format_usd(r.unit_price),
                "CapEx":              format_usd(r.capex_usd),
                f"{tco_amort}yr TCO": format_usd(r.tco_usd),
                "$/B-paths":          f"${r.cost_per_bpaths:,.4f}" if r.cost_per_bpaths else "—",
                "Rating":             r.rating or r.recommendation,
            })
            continue
        _span = (f"{r.gpus_per_copy} GPU / {r.nodes_per_copy} node"
                 + ("  ⚠multi-node" if r.nodes_per_copy > 1 else ""))
        tco_rows.append({
            "_result":            r,
            "System":             sname,
            "GPU":                sys_info["gpu_model"],
            "Link":               r.link_label,
            "BW/GPU":             f"{r.mem_bw_tbs:.2f} TB/s",
            "Decode Tok/s/User":  f"{r.tps_per_user:,.0f}" if r.predicted_tps else "—",
            "Per-Copy":           _span if r.predicted_tps else "—",
            "Nodes":              f"{r.num_nodes:,}",
            "GPUs":               f"{r.gpus_total:,}",
            "Unit $":             format_usd(r.unit_price),
            "CapEx":              format_usd(r.capex_usd),
            f"{tco_amort}yr TCO": format_usd(r.tco_usd),
            "$/MTok":             f"${r.cost_per_mtok:.4f}" if r.cost_per_mtok else "—",
            "Rating":             r.rating or r.recommendation,
        })

    # Ranking. LLM profile ranks by lowest total cost (TCO); FinTech profile ranks
    # by the bandwidth+capacity rating score (highest first) so the fastest-memory
    # platform leads. Either way Not-Viable systems sink to the bottom and error
    # rows go last.
    def _row_sortkey(row):
        r = row["_result"]
        if r is None:
            return (2, 0.0, 0.0)
        if r.rating == "Not Viable" or r.predicted_tps <= 0:
            return (1, 0.0, 0.0)
        if tco_profile == "fintech":
            # Bandwidth-bound MC sizing answer: fewest GPUs to hit the target, then
            # lowest TCO. High-BW parts (B200/GB300) need far fewer GPUs than Dell Pro Max GB10.
            return (0, r.gpus_total, r.tco_usd)
        return (0, r.tco_usd, -r.tps_per_user)
    tco_rows.sort(key=_row_sortkey)

    # -----------------------------------------------------------------------
    # Summary table — colour-coded combined Rating (Best → Not Viable)
    # -----------------------------------------------------------------------
    st.markdown("### System Comparison")
    if tco_profile == "fintech":
        st.caption(
            "Sized for a **bandwidth-bound Monte-Carlo** workload — ranked by **fewest GPUs** "
            "to hit your target paths/sec, then lowest TCO (Not-Viable last). **GPUs Needed** = "
            "`max(fit the working set, meet the demanded bandwidth)`; high-HBM parts (HBM3e ≫ "
            "GDDR7 ≫ LPDDR5X) need far fewer than Dell Pro Max GB10's 273 GB/s. **Rating** = per-GPU memory "
            "speed (70%) + VRAM capacity (30%), cost-independent. **Paths/s** is the achievable "
            f"throughput from the provisioned GPUs; **$/B-paths** is cost per billion paths over the "
            f"{tco_amort}-yr window. **Unit $** is the per-node/per-rack list price."
        )
    else:
        st.caption(
            f"Ranked by **total cost** — lowest {tco_amort}yr TCO first (Not-Viable systems last). "
            "**Rating** = performance + cost blend (Best → Not Viable). "
            "**Decode Tok/s/User** is the single-stream speed each user feels — set by per-GPU "
            "**memory bandwidth** (HBM3 ≫ HBM2) and the **Link** fabric used to span the model "
            "(NVLink ≫ PCIe; crossing nodes over InfiniBand costs more still). "
            "**Per-Copy** shows GPUs × nodes one model copy spans — ⚠multi-node means it can't fit "
            "one node and pays a cross-node penalty. **Unit $** is the per-node/per-rack list price."
        )
    # Auto-hide Not Viable: a system that cannot run the workload is noise in the comparison
    # (and at wide scopes it's most of the table). Hidden by default, one click to inspect.
    def _is_viable(row) -> bool:
        r = row.get("_result")
        return bool(r) and r.rating != "Not Viable" and not str(row.get("Rating", "")).startswith("Error")

    _viable_rows = [r for r in tco_rows if _is_viable(r)]
    _hidden_n    = len(tco_rows) - len(_viable_rows)
    _show_nv     = False
    if _hidden_n:
        _show_nv = st.checkbox(
            f"Show {_hidden_n} not-viable system{'s' if _hidden_n != 1 else ''}",
            value=False, key=f"tco_show_nv_{tco_scope}",
            help="Systems that cannot run this workload (won't fit, or no viable config). "
                 "Hidden by default.",
        )
    _table_rows = tco_rows if _show_nv else _viable_rows
    if not _table_rows:
        st.warning(f"No viable systems for this workload — all {len(tco_rows)} were rated Not Viable. "
                   "Tick the box above to see why, or widen the scope / lower the precision.")
        st.stop()

    display_cols = [k for k in _table_rows[0].keys() if k != "_result"]
    df_tco = pd.DataFrame([{k: row[k] for k in display_cols} for row in _table_rows])

    def _color_rating(val):
        key = val if val in RATING_COLORS else (
            "Not Viable" if str(val).startswith("Error") else None)
        c = RATING_COLORS.get(key, "#333")
        return f"color: {c}; font-weight: bold"

    styled = df_tco.style.map(_color_rating, subset=["Rating"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Best-fit callout (top of the ranked list)
    # -----------------------------------------------------------------------
    ranked = [row for row in tco_rows if row["_result"] and row["_result"].rating != "Not Viable"]
    if ranked:
        best = ranked[0]
        br = best["_result"]
        if tco_profile == "fintech":
            st.success(
                f"🏆 **Fewest GPUs for the Monte-Carlo target: {best['System']}**  \n"
                f"### 📈 {br.gpus_total:,} GPUs · {br.paths_per_sec/1e9:,.2f}B paths/s\n"
                f"{br.mem_bw_tbs:.2f} TB/s/GPU · {br.working_set_gb:,.1f} GB working set · "
                f"{br.num_nodes} node(s)  \n"
                f"Highest usable memory bandwidth → fewest GPUs to hit the throughput target "
                f"for memory-bound Monte-Carlo risk / pricing.  \n"
                f"{tco_amort}-yr TCO: **{format_usd(br.tco_usd)}**"
                + (f" · **${br.cost_per_bpaths:,.4f}/B-paths**" if br.cost_per_bpaths else "")
                + f"  \n_{br.rec_reason}_"
            )
        else:
            st.success(
                f"🏆 **Recommended: {best['System']}**  \n"
                f"### 💰 Total Cost: {format_usd(br.tco_usd)}\n"
                f"{tco_amort}-yr TCO — {format_usd(br.capex_usd)} CapEx + power · "
                f"{br.num_nodes} node(s) · {br.gpus_total:,} GPUs  \n"
                + (f"Cost/Employee: **{format_usd(br.tco_usd / tco_employees)}** · "
                   if tco_employees else "")
                + f"Cost/Session: **{format_usd(br.cost_per_user)}** · "
                f"Perf: **{br.predicted_tps:,.0f} tok/s** ({br.tps_per_user:,.0f}/session)  \n"
                f"_{br.rating_reason}_"
            )
    else:
        st.error("No selected system can serve this workload at the chosen settings — all Not Viable.")

    # -----------------------------------------------------------------------
    # AI analysis (Claude) — features 2 (recommendation) & 3 (cloud comparison)
    # -----------------------------------------------------------------------
    _ai_ok, _ai_reason = narrator_available()
    if is_fintech:
        _workload = {
            "workload": "Monte-Carlo (bandwidth-bound quant)",
            "resident_paths": int(mc_resident), "bytes_per_path": int(mc_bytes),
            "timesteps": int(mc_steps), "target_paths_per_sec": int(mc_target),
            "amort_years": int(tco_amort), "power": float(tco_power),
        }
    else:
        _workload = {
            "model": tco_model, "precision": tco_precision, "users": int(tco_users),
            "employees": int(tco_employees),
            "output_toks": int(tco_output_toks), "context": int(tco_ctx),
            "amort_years": int(tco_amort), "power": float(tco_power),
        }
    _ai_systems, _best_entry = [], None
    for row in tco_rows:  # already sorted best → worst
        r = row.get("_result")
        if r is None:
            continue
        if r.rating != "Not Viable":
            _ai_systems.append({
                "name": row["System"], "rating": r.rating or r.recommendation,
                "decode_per_user": r.tps_per_user, "cluster_tps": r.predicted_tps,
                "mem_bw_tbs": r.mem_bw_tbs, "link": r.link_label,
                "nodes": r.num_nodes, "gpus": r.gpus_total, "amort": tco_amort,
                "tco": format_usd(r.tco_usd),
                "cost_per_mtok": (f"${r.cost_per_mtok:.4f}" if r.cost_per_mtok else "n/a"),
                "mc": is_fintech, "paths_per_sec": r.paths_per_sec,
                "cost_per_bpaths": (f"${r.cost_per_bpaths:.4f}" if r.cost_per_bpaths else "n/a"),
            })
            # First viable row is the top-ranked "Best" platform — cloud compares to it.
            if _best_entry is None:
                _best_entry = {
                    "name": row["System"],
                    "decode_per_user": r.tps_per_user,
                    "cost_per_mtok": (f"${r.cost_per_mtok:.4f}" if r.cost_per_mtok else "n/a"),
                    "tco": format_usd(r.tco_usd),
                    "mem_bw_tbs": r.mem_bw_tbs, "gpus": r.gpus_total, "nodes": r.num_nodes,
                    "fits": True,
                }

    # Hosted-API token cost comparison — only meaningful for the token-based LLM
    # profile (FinTech Monte-Carlo has no token notion).
    if not is_fintech:
        _api_costs = api_token_costs(int(tco_ctx), int(tco_output_toks))

        with st.expander("☁️ Cloud API token cost for this workload (Claude / Gemini)", expanded=False):
            _best_name = _best_entry["name"] if _best_entry else "the best on-prem system"
            _best_mtok = _best_entry["cost_per_mtok"] if _best_entry else "n/a"
            st.caption(
                f"Per request = {tco_ctx:,} input + {tco_output_toks:,} output tokens. "
                f"**$/1M output tok** loads the proportional input, so it compares directly to the "
                f"best on-prem option (**{_best_name}: {_best_mtok} per 1M output tok**)."
            )
            st.dataframe(pd.DataFrame([
                {
                    "Model":            c["model"],
                    "Provider":         c["provider"],
                    "$/Request":        f"${c['cost_per_request']:.5f}",
                    "$/1M output tok":  f"${c['cost_per_mtok_out']:,.2f}",
                    "List in/out $/1M": f"${c['in_price']:.2f} / ${c['out_price']:.2f}",
                }
                for c in _api_costs
            ]), use_container_width=True, hide_index=True)
            st.caption("Published standard-context list prices (2026). Sources: platform.claude.com, "
                       "ai.google.dev/gemini-api/docs/pricing.")

    st.markdown("### 🟢 On-Device AI Analysis")
    st.caption("Generated locally on the Dell Pro Max GB10 by "
               f"{NARRATOR_MODEL.split('/')[-1]} — no cloud."
               if _ai_ok else f"⚠ {_ai_reason}")
    # Cloud-API token comparison only applies to the token-based LLM profile.
    if is_fintech:
        _do_rec = st.button("🧠 AI Recommendation (on-device)", key="ai_tco_rec",
                            use_container_width=True, disabled=not _ai_ok)
        _do_cloud = False
    else:
        _aic1, _aic2 = st.columns(2)
        _do_rec = _aic1.button("🧠 AI Recommendation (on-device)", key="ai_tco_rec",
                               use_container_width=True, disabled=not _ai_ok)
        _do_cloud = _aic2.button("☁️ Compare to Cloud APIs (on-device)", key="ai_tco_cloud",
                                 use_container_width=True, disabled=not _ai_ok)
    if _do_rec or _do_cloud:
        _narr = ensure_narrator()   # resident
        if _narr and _narr.ready:
            if _do_rec:
                st.write_stream(_narr.tco_recommendation(_workload, _ai_systems[:8]))
            else:
                st.write_stream(_narr.cloud_comparison(_workload, _best_entry, _api_costs))
        else:
            st.warning(_narr.reason if _narr else _ai_reason)

    # -----------------------------------------------------------------------
    # Bar charts — CapEx and 3yr TCO side-by-side
    # -----------------------------------------------------------------------
    chart_rows = [r for r in tco_rows if r["_result"] and r["_result"].rating != "Not Viable"]
    if chart_rows:
        st.markdown("### Cost Breakdown")
        cc1, cc2 = st.columns(2)

        # Bars are coloured by GPU ARCHITECTURE FAMILY, not per system: at ~76 systems a
        # per-system hue would be meaningless, and a categorical palette must never exceed
        # ~8 hues nor be cycled. Falls back to muted ink rather than inventing a hue.
        def _sys_color(name: str) -> str:
            return DELL_SYSTEMS.get(name, {}).get("color", "#898781")

        # Unique, readable axis label — chassis on line 1, GPU config on line 2.
        # Most chassis carry several GPU options (R770 alone has 6), so a label must be
        # chassis + GPU or the rows collide. Read the catalog's own `platform`/`gpu_label`
        # fields rather than parsing the key — parsing breaks on names that contain nested
        # parens, e.g. "XE9780L (GNR AP) (B300 PC (x8))".
        def _chart_label(name: str) -> str:
            si = DELL_SYSTEMS.get(name, {})
            base = si.get("platform") or (
                name.split("(")[0].strip().replace("Dell PowerEdge ", "").replace("Dell ", ""))
            gpu = si.get("gpu_label", "")
            return f"{base}<br>{gpu}" if gpu else base

        sys_labels = [_chart_label(r["System"]) for r in chart_rows]
        capex_vals = [r["_result"].capex_usd for r in chart_rows]
        tco_vals   = [r["_result"].tco_usd for r in chart_rows]
        colors_bar = [_sys_color(r["System"]) for r in chart_rows]

        with cc1:
            fig_capex = go.Figure(go.Bar(
                x=sys_labels, y=capex_vals,
                marker_color=colors_bar, text=[format_usd(v) for v in capex_vals],
                textposition="outside",
            ))
            fig_capex.update_layout(
                title="CapEx (with infra overhead)", yaxis_title="USD",
                plot_bgcolor="#F5F7FA", paper_bgcolor="#F5F7FA",
                font=dict(color="#1C2B4B"),
                height=350, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_capex, use_container_width=True)

        with cc2:
            fig_tco = go.Figure(go.Bar(
                x=sys_labels, y=tco_vals,
                marker_color=colors_bar, text=[format_usd(v) for v in tco_vals],
                textposition="outside",
            ))
            fig_tco.update_layout(
                title=f"{tco_amort}-Year TCO (CapEx + Power)", yaxis_title="USD",
                plot_bgcolor="#F5F7FA", paper_bgcolor="#F5F7FA",
                font=dict(color="#1C2B4B"),
                height=350, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_tco, use_container_width=True)

        # $/MTok comparison
        mtok_rows = [r for r in chart_rows if r["_result"].cost_per_mtok]
        if mtok_rows:
            st.markdown("### Cost Efficiency — $/Million Tokens")
            mtok_labels = [_chart_label(r["System"]) for r in mtok_rows]
            mtok_vals   = [r["_result"].cost_per_mtok for r in mtok_rows]
            mtok_colors = [_sys_color(r["System"]) for r in mtok_rows]

            fig_mtok = go.Figure(go.Bar(
                x=mtok_labels, y=mtok_vals,
                marker_color=mtok_colors,
                text=[f"${v:.4f}" for v in mtok_vals], textposition="outside",
            ))
            fig_mtok.update_layout(
                title="Cost per Million Tokens Generated",
                yaxis_title="$ / MTok",
                plot_bgcolor="#F5F7FA", paper_bgcolor="#F5F7FA",
                font=dict(color="#1C2B4B"),
                height=320, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_mtok, use_container_width=True)

    # -----------------------------------------------------------------------
    # Per-system detail cards
    # -----------------------------------------------------------------------
    st.markdown("### System Details")
    # Each card is an expander holding 12 st.metric widgets. Rendering the whole catalog would
    # be ~900 widgets on every rerun, so cap to the best N (tco_rows is already sorted).
    _DETAIL_CAP = 12
    # Follows the same not-viable filter as the table above.
    _detail_rows = [r for r in _table_rows if r.get("_result") is not None]
    if len(_detail_rows) > _DETAIL_CAP:
        st.caption(f"Showing details for the top {_DETAIL_CAP} of {len(_detail_rows)} systems "
                   "(best-rated first). Narrow the scope or selection to see others.")
        _detail_rows = _detail_rows[:_DETAIL_CAP]
    for row in _detail_rows:
        r = row["_result"]
        sys_info = DELL_SYSTEMS[row["System"]]
        rate_color = RATING_COLORS.get(r.rating, "#333")

        with st.expander(f"{row['System']}  —  {r.rating or r.recommendation}", expanded=False):
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("CapEx (incl. infra)", format_usd(r.capex_usd))
            d2.metric(f"{tco_amort}yr TCO", format_usd(r.tco_usd))
            d3.metric("Annual Power Cost", format_usd(r.annual_power_usd))
            if is_fintech:
                d4.metric("$ / Billion paths",
                          f"${r.cost_per_bpaths:,.4f}" if r.cost_per_bpaths else "—",
                          help=f"Over the {tco_amort}-yr window at full utilization")
            else:
                d4.metric("Cost / Employee" if tco_employees else "Cost / Session",
                          format_usd(r.tco_usd / tco_employees) if tco_employees
                          else format_usd(r.cost_per_user),
                          help=(f"{tco_amort}-yr TCO ÷ {tco_employees:,} employees · "
                                f"{format_usd(r.cost_per_user)} per effective session")
                               if tco_employees else None)

            d5, d6, d7, d8 = st.columns(4)
            d5.metric("Total Nodes / GPUs", f"{r.num_nodes:,} / {r.gpus_total:,}",
                      help="Whole solution sized for the workload")
            if is_fintech:
                d6.metric("Throughput", f"{r.paths_per_sec/1e9:,.2f}B paths/s",
                          help="Achievable Monte-Carlo paths/sec from the provisioned GPUs")
            else:
                d6.metric("Decode Tok/s per User", f"{r.tps_per_user:,.0f}",
                          help="Single-stream speed each user experiences")
            d7.metric("Mem BW / GPU", f"{r.mem_bw_tbs:.2f} TB/s")
            d8.metric("Unit List Price", format_usd(r.unit_price),
                      help="Per node, or per 72-GPU rack for NVL72")

            d9, d10, d11, d12 = st.columns(4)
            d9.metric("GPU Link", r.link_label or "—")
            d10.metric("GPUs per Copy", r.gpus_per_copy)
            d11.metric("Nodes per Copy", r.nodes_per_copy,
                       help=">1 means the model spans nodes over InfiniBand (cross-node penalty)")
            d12.metric("Parallel Efficiency", f"{r.parallel_eff:.0%}",
                       help="Throughput kept after tensor-parallel + cross-node communication loss")

            st.markdown(
                f"<span style='color:{rate_color};font-weight:bold'>● {r.rating or r.recommendation}</span>"
                f" — {r.rating_reason or r.rec_reason}",
                unsafe_allow_html=True,
            )
            st.caption(sys_info["notes"])
            best_for = ", ".join(sys_info.get("best_for", []))
            if best_for:
                st.caption(f"Best for: {best_for}")

    # -----------------------------------------------------------------------
    # Dell Pro Max GB10 vs scale-up comparison (if benchmark data exists)
    # -----------------------------------------------------------------------
    if _gb10_tps_measured and not is_fintech:
        st.divider()
        st.markdown("### Dell Pro Max GB10 Measured → Scale-Up Projection")
        st.caption(
            f"Measured Dell Pro Max GB10 throughput: **{_gb10_tps_measured:.0f} tok/s** — "
            "projected to larger systems using memory-bandwidth ratio scaling."
        )
        from helpers.tco_engine import scale_throughput, gpus_needed_for_model
        scale_rows = []
        # Follow the selection — this used to iterate the whole catalog, which at ~76 systems
        # ignores the scope entirely and renders every row regardless of what's being compared.
        for sname in tco_systems:
            sinfo = DELL_SYSTEMS[sname]
            _gpus_needed = gpus_needed_for_model(m_gb, sinfo["vram_gb"] / max(sinfo["gpus_per_node"], 1))
            _proj_tps = scale_throughput(
                _gb10_tps_measured, sinfo["gpu_bw_gbs"],
                _gpus_needed, GB10_BW_GBS,
            )
            scale_rows.append({
                # Full name, not split("(")[0] — that collapsed every GPU variant of a chassis
                # into identical rows (all 6 R770 builds became one "Dell PowerEdge R770").
                "System":       sname.replace("Dell PowerEdge ", "").replace("Dell ", ""),
                "BW (GB/s)":    f"{sinfo['gpu_bw_gbs']:,}",
                "BW vs Dell Pro Max GB10":   f"{sinfo['gpu_bw_gbs'] / GB10_BW_GBS:.1f}×",
                "Proj. TPS":    f"{_proj_tps:,.0f}",
                "Speedup":      f"{_proj_tps / max(_gb10_tps_measured, 1):.1f}×",
            })
        st.dataframe(pd.DataFrame(scale_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "<div class='dell-footer'>"
    "<span>DELL</span> technologies &nbsp;·&nbsp; Dell Pro Max GB10 Demo Suite "
    "&nbsp;·&nbsp; Blackwell · Dell Pro Max GB10 · aarch64 "
    "&nbsp;·&nbsp; Confidential — For Internal Sales Use Only"
    "</div>",
    unsafe_allow_html=True,
)
