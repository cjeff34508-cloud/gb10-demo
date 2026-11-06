#!/usr/bin/env python3
"""Smoke tests for the TCO capacity model (replicated vs shared-weights).

Standalone — no pytest. Run from anywhere:

    python scripts/smoke_tco_capacity.py            # run all applicable assertions
    python scripts/smoke_tco_capacity.py --generate # (re)write the golden baseline

The script is checkpoint-aware: assertions that need the new `capacity_mode` API
or the new `TCOResult` fields are skipped (with a note) until those land, so the
same file is green at every checkpoint of the shared-weights rollout.

Assertions
  1. Back-compat  — replicated-mode output == golden baseline (within 1e-6).
  2. Ordering     — shared sessions/node >= replicated copies/node (every LLM).
  3. Cost sanity  — shared nodes <= replicated nodes; shared $/MTok <= replicated.
  4. Monotonicity — shared sessions/node non-increasing as context grows.
  5. Captures     — analytic shared aggregate within [0.5, 2.0]x the real GB10
                    vLLM captures (webui/data/vllm_reference.json).
  6. Edge cases   — a >1-GPU model still sizes via TP/multi-node; no div-by-zero.
  7. Toggle       — capacity_mode actually moves platform sizing.
  8. KV geometry  — KV/session depends on attention geometry, NOT expert count
                    or total params. REGRESSION TEST: red before the C3 fix.
  9. KV linear    — KV/session is linear in context and zero at zero context.
 10. Capacity     — weights_once still scales with TOTAL params, never active_b.
 11. MLA KV       — an MLA model's KV/session is below a GQA model of equal depth.
                    REGRESSION TEST: red before the C3 fix.
 12. Calibration  — the KV formula reproduces the two KV figures MEASURED on
                    this box: 48.0 KiB/tok (Qwen3-Coder-30B-A3B) and 10.0
                    KiB/tok (Qwen3.6-35B-A3B), both at fp8.
"""
from __future__ import annotations
import argparse
import inspect
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "webui"))

from helpers.tco_engine import (  # noqa: E402
    calculate_tco, MODEL_CATALOG, native_precision,
)

BASELINE = os.path.join(HERE, "tco_golden_baseline.json")
VLLM_REF = os.path.join(ROOT, "webui", "data", "vllm_reference.json")

# --- capability detection: which parts of the new API exist yet -------------
_SIG = inspect.signature(calculate_tco)
HAS_MODE = "capacity_mode" in _SIG.parameters

# --- fixed test matrix ------------------------------------------------------
MODELS  = ["TinyLlama-1.1B", "Qwen2.5-7B", "Llama-3.3-70B",
           "Mixtral-8x7B", "Llama-3.1-405B"]
SYSTEMS = ["Dell Pro Max GB10",
           "Dell PowerEdge XE9680 (H200 SXM5 (x8) AC)",
           "Dell PowerEdge XE7740 (RTX Pro 6000 BSE)"]
USERS   = [1, 8, 64]
CTXS    = [512, 131072]
GB10_TPS = 1200.0
OUTPUT_TOKS = 512

# Fields captured for the back-compat baseline (numeric compared @1e-6, strings exact).
_NUM_FIELDS = ("num_nodes", "capex_usd", "annual_power_usd", "tco_usd",
               "cost_per_user", "cost_per_mtok", "predicted_tps", "gpu_util_pct",
               "tps_per_user", "bw_contention", "fleet_overhead_pct",
               "gpus_per_node", "gpus_per_copy", "nodes_per_copy", "gpus_total",
               "unit_price")
_STR_FIELDS = ("recommendation", "rec_reason")


def _call(model, system, users, ctx, mode=None, gb10_tps=GB10_TPS):
    kw = dict(system_name=system, model_name=model, precision=native_precision(model),
              num_users=users, gb10_tps=gb10_tps, output_toks=OUTPUT_TOKS,
              context_len=ctx, amort_years=3, power_rate=0.12, add_infra_pct=0.15)
    if HAS_MODE and mode is not None:
        kw["capacity_mode"] = mode
    return calculate_tco(**kw)


def _key(model, system, users, ctx):
    return f"{model}|{system}|{users}|{ctx}"


def _snapshot(r) -> dict:
    d = {f: getattr(r, f) for f in _NUM_FIELDS}
    d.update({f: getattr(r, f) for f in _STR_FIELDS})
    return d


def generate_baseline():
    """Snapshot current replicated-mode output across the matrix."""
    out = {}
    for m in MODELS:
        for s in SYSTEMS:
            for u in USERS:
                for c in CTXS:
                    out[_key(m, s, u, c)] = _snapshot(_call(m, s, u, c, mode="replicated"))
    with open(BASELINE, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"■ wrote golden baseline: {BASELINE}  ({len(out)} rows)")


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
class Fail(AssertionError):
    pass


def _close(a, b, tol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol + tol * max(abs(a), abs(b))


def a1_backcompat(report):
    if not os.path.exists(BASELINE):
        report("1 back-compat", None, "no baseline file — run --generate first")
        return
    base = json.load(open(BASELINE))
    bad = []
    for k, exp in base.items():
        m, s, u, c = k.split("|")
        got = _snapshot(_call(m, s, int(u), int(c), mode="replicated"))
        for f in _NUM_FIELDS:
            if not _close(got[f], exp[f]):
                bad.append(f"{k}: {f} {exp[f]} != {got[f]}")
        for f in _STR_FIELDS:
            if got[f] != exp[f]:
                bad.append(f"{k}: {f} {exp[f]!r} != {got[f]!r}")
    if bad:
        raise Fail("replicated output drifted from baseline:\n  " + "\n  ".join(bad[:8]))
    report("1 back-compat", True, f"{len(base)} rows identical to baseline")


def _llm_models():
    return [m for m, i in MODEL_CATALOG.items() if i.get("category") == "LLM"]


def a2_ordering(report):
    if not HAS_MODE:
        report("2 ordering", None, "capacity_mode not implemented yet")
        return
    bad = []
    for m in _llm_models():
        for s in ("Dell Pro Max GB10",):
            rr = _call(m, s, 64, 512, mode="replicated")
            rs = _call(m, s, 64, 512, mode="shared")
            if rr.num_nodes == 0 or rs.num_nodes == 0:
                continue  # not-viable on this system; skip
            if rs.sessions_per_node < rr.sessions_per_node:
                bad.append(f"{m}: shared {rs.sessions_per_node} < replicated {rr.sessions_per_node}")
    if bad:
        raise Fail("shared should never fit FEWER sessions/node:\n  " + "\n  ".join(bad[:8]))
    report("2 ordering", True, "shared sessions/node >= replicated for all viable LLMs")


def a3_cost(report):
    if not HAS_MODE:
        report("3 cost", None, "capacity_mode not implemented yet")
        return
    bad = []
    for m in ("Qwen2.5-7B", "TinyLlama-1.1B", "Llama-3.3-70B"):
        rr = _call(m, "Dell Pro Max GB10", 64, 512, mode="replicated")
        rs = _call(m, "Dell Pro Max GB10", 64, 512, mode="shared")
        if rr.num_nodes == 0 or rs.num_nodes == 0:
            continue
        if rs.num_nodes > rr.num_nodes:
            bad.append(f"{m}: shared nodes {rs.num_nodes} > replicated {rr.num_nodes}")
        if rs.cost_per_mtok and rr.cost_per_mtok and rs.cost_per_mtok > rr.cost_per_mtok * (1 + 1e-6):
            bad.append(f"{m}: shared $/MTok {rs.cost_per_mtok:.2f} > replicated {rr.cost_per_mtok:.2f}")
    if bad:
        raise Fail("shared should cost no more than replicated:\n  " + "\n  ".join(bad))
    report("3 cost", True, "shared nodes & $/MTok <= replicated")


def a4_monotonic(report):
    if not HAS_MODE:
        report("4 monotonic", None, "capacity_mode not implemented yet")
        return
    bad = []
    ctxs = [512, 4096, 32768, 131072, 262144]
    for m in ("Qwen2.5-7B", "Llama-3.3-70B"):
        prev = None
        for c in ctxs:
            r = _call(m, "Dell Pro Max GB10", 8, c, mode="shared")
            spn = r.sessions_per_node
            if prev is not None and spn > prev:
                bad.append(f"{m}: sessions/node rose {prev}->{spn} at ctx {c}")
            prev = spn
    if bad:
        raise Fail("shared sessions/node must not rise with context:\n  " + "\n  ".join(bad))
    report("4 monotonic", True, "shared sessions/node non-increasing as context grows")


# maps catalog names -> vllm_reference capture keys
_CAPTURE_MAP = {"Qwen2.5-7B": "qwen2.5-7b", "nvidia/Qwen3-8B-NVFP4": "qwen3-8b-nvfp4"}


def a5_captures(report):
    if not HAS_MODE:
        report("5 captures", None, "capacity_mode not implemented yet")
        return
    if not os.path.exists(VLLM_REF):
        report("5 captures", None, "no vllm_reference.json")
        return
    caps = json.load(open(VLLM_REF)).get("captures", {})
    bad = []
    checked = 0
    for model, ckey in _CAPTURE_MAP.items():
        cap = caps.get(ckey)
        if not cap or model not in MODEL_CATALOG:
            continue
        rows = cap.get("per_n", [])
        # Anchor the single-user baseline to the capture's own n=1 measurement
        # (analytic aggregate = n * baseline * contention; the exponent under test).
        base_row = next((x for x in rows if x["n_users"] == 1), None)
        baseline_tps = (base_row or rows[0])["per_user_tps"]
        for row in rows:
            n = row["n_users"]
            if n < 8:
                continue
            r = _call(model, "Dell Pro Max GB10", n, 512, mode="shared", gb10_tps=baseline_tps)
            if r.num_nodes == 0:
                continue
            analytic = r.predicted_tps            # aggregate tok/s
            captured = row["aggregate_tps"]
            ratio = analytic / max(captured, 1e-9)
            checked += 1
            if not (0.5 <= ratio <= 2.0):
                bad.append(f"{model} n={n}: analytic {analytic:.0f} vs captured {captured:.0f} (ratio {ratio:.2f})")
    if bad:
        raise Fail("shared aggregate outside [0.5,2.0]x real captures:\n  " + "\n  ".join(bad))
    report("5 captures", True, f"{checked} capture points within band [0.5,2.0]x")


def a7_toggle(report):
    if not HAS_MODE:
        report("7 toggle", None, "capacity_mode not implemented yet")
        return
    # The toggle must actually move platform sizing for a small model on the GB10.
    rr = _call("Qwen2.5-7B", "Dell Pro Max GB10", 64, 512, mode="replicated")
    rs = _call("Qwen2.5-7B", "Dell Pro Max GB10", 64, 512, mode="shared")
    if not (rs.num_nodes < rr.num_nodes and rs.sessions_per_node > rr.sessions_per_node):
        raise Fail(f"toggle had no sizing effect: replicated nodes={rr.num_nodes}/"
                   f"sess={rr.sessions_per_node} vs shared nodes={rs.num_nodes}/sess={rs.sessions_per_node}")
    report("7 toggle", True, f"Qwen2.5-7B: {rr.num_nodes} nodes (repl) -> {rs.num_nodes} node(s) (shared)")


def a6_edges(report):
    # A model too big for one GPU must still size via TP / multi-node in BOTH modes.
    modes = (None,) if not HAS_MODE else ("replicated", "shared")
    bad = []
    for mode in modes:
        r = _call("Llama-3.1-405B", "Dell PowerEdge XE9680 (H200 SXM5 (x8) AC)", 4, 4096, mode=mode)
        if r.num_nodes > 0 and r.gpus_per_copy < 2:
            bad.append(f"405B mode={mode}: expected multi-GPU copy, got gpus_per_copy={r.gpus_per_copy}")
        # never NaN/inf from a divide-by-zero
        for f in ("tco_usd", "predicted_tps", "cost_per_user"):
            v = getattr(r, f)
            if v is not None and (math.isnan(v) or math.isinf(v)):
                bad.append(f"405B mode={mode}: {f} is {v}")
    if bad:
        raise Fail("edge cases:\n  " + "\n  ".join(bad))
    report("6 edges", True, "405B sizes via TP/multi-node; no NaN/inf")


# ---------------------------------------------------------------------------
# MoE-awareness invariants (C4). These do NOT pin numbers — they pin RELATIONS
# that must hold whatever the KV model is, so they survive a deliberate
# re-baseline while still catching a wrong KV model.
#
# a8 and a11 are REGRESSION TESTS: they fail against the weights-relative
# `_KV_FRAC_PER_TOKEN` heuristic and pass once KV comes from attention
# geometry. A guard that was green before the fix would not have been a guard.
# ---------------------------------------------------------------------------
from contextlib import contextmanager  # noqa: E402
from helpers.tco_engine import MODEL_ARCH, session_footprint  # noqa: E402

# Identical attention geometry (so identical KV), wildly different expert count
# and therefore total parameters. 32 layers x 8 kv_heads x 4096 hidden.
_GEOM = {"layers": 32, "q_heads": 32, "kv_heads": 8, "hidden": 4096,
         "ctx": 131072, "attn": "GQA"}
_SYNTH = {
    "__T-moe-small": (
        {"params_b": 47.0, "category": "LLM", "type": "moe", "active_b": 13.0},
        dict(_GEOM, experts=8, topk=2),
    ),
    "__T-moe-huge": (   # 5x the total params, SAME attention -> SAME KV/session
        {"params_b": 235.0, "category": "LLM", "type": "moe", "active_b": 22.0},
        dict(_GEOM, experts=128, topk=8),
    ),
    "__T-dense-gqa": (
        {"params_b": 47.0, "category": "LLM", "type": "decoder"},
        dict(_GEOM),
    ),
    "__T-mla": (        # MLA: KV is a 512-wide latent, not kv_heads x head_dim
        {"params_b": 671.0, "category": "LLM", "type": "moe", "active_b": 37.0},
        dict(_GEOM, attn="MLA", kv_latent=512, experts=256, topk=8),
    ),
}


@contextmanager
def _synthetic_models():
    """Inject the probe models, then remove them. In-process only."""
    for name, (cat, arch) in _SYNTH.items():
        MODEL_CATALOG[name] = cat
        MODEL_ARCH[name] = arch
    try:
        yield
    finally:
        for name in _SYNTH:
            MODEL_CATALOG.pop(name, None)
            MODEL_ARCH.pop(name, None)


def _kv(model, ctx=32768, prec="FP8", batch=1):
    return session_footprint(model, prec, context_len=ctx, batch_size=batch)[1]


def a8_kv_expert_independent(report):
    """KV-per-session must depend on ATTENTION GEOMETRY, not on expert count.

    Two MoEs with identical layers/kv_heads/hidden must get identical KV even
    though one has 5x the total parameters. This is the C3 defect exactly:
    `session_footprint` derives KV from `params_b x bytes_per_param`.
    """
    with _synthetic_models():
        small = _kv("__T-moe-small")
        huge  = _kv("__T-moe-huge")
    if small <= 0:
        raise Fail(f"KV-per-session is {small} — expected > 0")
    ratio = huge / small
    if not _close(ratio, 1.0, tol=0.02):
        raise Fail(
            f"KV scales with total params, not attention geometry: "
            f"47B-A13 -> {small:.3f} GB vs 235B-A22 -> {huge:.3f} GB "
            f"(ratio {ratio:.2f}, expected 1.00). Same layers/kv_heads/hidden "
            f"must mean same KV; only expert count differs.")
    report("8 kv-geometry", True,
           f"KV/session identical across 5x expert inflation ({small:.3f} GB)")


def a9_kv_linear_in_context(report):
    """KV-per-session must be linear in context and zero at zero context."""
    with _synthetic_models():
        z  = _kv("__T-dense-gqa", ctx=0)
        k1 = _kv("__T-dense-gqa", ctx=8192)
        k2 = _kv("__T-dense-gqa", ctx=16384)
    bad = []
    if not _close(z, 0.0, tol=1e-9):
        bad.append(f"ctx=0 gives {z} GB of KV, expected 0")
    if k1 <= 0:
        bad.append(f"ctx=8192 gives {k1} GB, expected > 0")
    elif not _close(k2 / k1, 2.0, tol=0.02):
        bad.append(f"doubling context scaled KV by {k2/k1:.3f}x, expected 2.00x")
    if bad:
        raise Fail("KV must be linear in context:\n  " + "\n  ".join(bad))
    report("9 kv-linear", True,
           f"KV 0 at ctx=0, {k1:.3f}->{k2:.3f} GB on a 2x context")


def a10_capacity_uses_total(report):
    """Weights must still be sized from TOTAL params, never active params.

    An MoE occupies all its experts in VRAM even though it only reads a few per
    token. If a well-meaning 'MoE awareness' change made weights follow
    active_b, every capacity number on the box would become fiction.
    """
    with _synthetic_models():
        w_small = session_footprint("__T-moe-small", "FP8", context_len=512)[0]
        w_huge  = session_footprint("__T-moe-huge",  "FP8", context_len=512)[0]
    ratio = w_huge / max(w_small, 1e-9)
    if ratio < 4.0:
        raise Fail(
            f"weights_once for a 235B MoE is only {ratio:.2f}x a 47B MoE "
            f"({w_huge:.1f} vs {w_small:.1f} GB) — expected ~5x (total params). "
            f"Weights must NOT follow active_b.")
    report("10 capacity-total", True,
           f"weights scale with total params ({w_small:.0f} -> {w_huge:.0f} GB, {ratio:.1f}x)")


def a11_mla_cheaper_than_gqa(report):
    """MLA must yield LESS KV than GQA at the same layer count.

    DeepSeek-style MLA compresses KV to a `kv_latent`-wide vector per layer.
    512 latent vs 8 kv_heads x 128 head_dim = 1024 wide, so MLA must be cheaper
    per token despite the model being far larger in total.
    """
    with _synthetic_models():
        gqa = _kv("__T-dense-gqa")
        mla = _kv("__T-mla")
    if mla >= gqa:
        raise Fail(
            f"MLA KV/session {mla:.3f} GB >= GQA {gqa:.3f} GB at the same "
            f"32 layers. MLA's kv_latent=512 is narrower than 8 kv_heads x 128, "
            f"so it must be cheaper — this says KV is still following total "
            f"parameters (671B vs 47B) instead of attention geometry.")
    report("11 mla-kv", True,
           f"MLA KV/session {mla:.3f} GB < GQA {gqa:.3f} GB at equal depth")



def a12_kv_matches_box_measurements(report):
    """The KV formula must reproduce the two KV figures measured on this box.

    This is the calibration test. The relational assertions (8-11) prove the
    model has the right SHAPE; only these two numbers prove it has the right
    SCALE, and they are the only KV measurements anyone has taken here.

      * Qwen3-Coder-30B-A3B @ --kv-cache-dtype fp8 -> 48.0 KiB/token.
        qwen-hermes recorded "predicted 48.0 KB/token == observed".
      * Qwen3.6-35B-A3B @ fp8 -> 10.0 KiB/token, the figure gb10-markets
        recorded for the model it serves on :8504.

    Both are hybrids, so both also pin `attn_layers`: using the full layer
    count instead would give 96.0 and 40.0 KiB/token respectively.
    """
    from helpers.tco_engine import kv_bytes_per_token
    bad = []

    # Qwen3-Coder-30B-A3B is not in the catalog; probe its real shape directly
    # (48 layers all full-attention, 4 kv_heads, head_dim 128).
    with _synthetic_models():
        MODEL_ARCH["__T-coder30b"] = {"layers": 48, "q_heads": 32, "kv_heads": 4,
                                      "hidden": 2048, "head_dim": 128, "attn": "GQA"}
        MODEL_CATALOG["__T-coder30b"] = {"params_b": 30.0, "category": "LLM",
                                         "type": "moe", "active_b": 3.0}
        got = kv_bytes_per_token("__T-coder30b", "FP8")
        MODEL_ARCH.pop("__T-coder30b", None)
        MODEL_CATALOG.pop("__T-coder30b", None)
    if got is None or not _close(got / 1024.0, 48.0, tol=0.01):
        bad.append(f"Qwen3-Coder-30B-A3B shape @fp8: {(got or 0)/1024:.1f} KiB/tok, measured 48.0")

    got = kv_bytes_per_token("Qwen3.6-35B-A3B", "FP8")
    if got is None or not _close(got / 1024.0, 10.0, tol=0.01):
        bad.append(f"Qwen3.6-35B-A3B @fp8: {(got or 0)/1024:.1f} KiB/tok, measured 10.0 "
                   f"(40 layers but only 10 are full_attention — check attn_layers)")

    if bad:
        raise Fail("KV formula no longer matches on-box measurements:\n  " + "\n  ".join(bad))
    report("12 kv-calibration", True,
           "reproduces both measured figures: 48.0 and 10.0 KiB/token @fp8")


# ---------------------------------------------------------------------------
# G2b / G3 — the 2026 frontier entries and the PREDICTED anchor tier
# ---------------------------------------------------------------------------

# KV bytes/token at FP8, read off each repo's own config.json on 2026-11-06.
# A wrong `attn_layers` is the single likeliest error in a hybrid entry and it
# is silent — the model still sizes, just wrongly — so these are pinned exactly.
_G2B_KV_KIB_FP8 = {
    "Nemotron-3-Ultra-550B-A55B":   6.5,    # 13 attention layers of 108
    "Kimi-K3 (2.8T)":              13.5,    # 24 full MLA layers of 93
    "DeepSeek-V4-Pro (1.6T)":      34.3,    # 61 layers, 576-wide absorbed MLA
    "GLM-5.2-753B-A40B":           43.9,    # 78 layers, 576-wide absorbed MLA
    "Inkling-975B-A41B":          132.0,    # all 66 layers, upper bound
    "Inkling-975B-A41B (BF16)":   132.0,
}


def a13_frontier_kv_geometry(report):
    """The six frontier entries must reproduce their config-derived KV exactly.

    Each must also come from ATTENTION GEOMETRY, not the legacy
    fraction-of-weights fallback — `kv_bytes_per_token` returning None would
    mean a missing MODEL_ARCH row, and the fallback over-charges an MoE's KV
    wildly (that is the defect assertion 8 guards).
    """
    from helpers.tco_engine import kv_bytes_per_token
    for name, want in _G2B_KV_KIB_FP8.items():
        if name not in MODEL_CATALOG:
            raise Fail(f"{name} missing from MODEL_CATALOG")
        got = kv_bytes_per_token(name, "FP8")
        if got is None:
            raise Fail(f"{name}: kv_bytes_per_token is None — no MODEL_ARCH row, so "
                       f"KV silently reverts to the fraction-of-weights model")
        kib = got / 1024.0
        if abs(kib - want) > 0.1:
            raise Fail(f"{name}: KV {kib:.1f} KiB/tok, expected {want} "
                       f"(check `attn_layers` / `kv_latent` against config.json)")
    report("13 frontier-kv", True,
           f"{len(_G2B_KV_KIB_FP8)} frontier entries reproduce their config-derived KV "
           f"(6.5-132.0 KiB/tok, all geometric)")


def a14_frontier_cannot_fit_gb10(report):
    """Every frontier entry must fail to fit a GB10, at every precision it offers.

    These are projection-only by construction. If one ever reads as fitting, a
    `disk_gb` is wrong or a precision list grew — both of which would put a
    multi-hundred-GB model in front of a customer as desktop-viable.
    """
    from helpers.tco_engine import (model_memory_gb, supported_precisions,
                                    gpus_needed_for_model)
    from helpers.mem_guard import GB10_USABLE_GB
    worst = []
    for name in _G2B_KV_KIB_FP8:
        for prec in supported_precisions(name):
            gb = model_memory_gb(name, prec, batch_size=1, context_len=512)
            if gb <= GB10_USABLE_GB:
                raise Fail(f"{name} @{prec} reads as {gb:.1f} GB, inside the "
                           f"{GB10_USABLE_GB:g} GB usable window — it must not")
            n = gpus_needed_for_model(gb, GB10_USABLE_GB)
            if n < 2:
                raise Fail(f"{name} @{prec} needs {n} GB10 for {gb:.1f} GB")
            worst.append(n)
    report("14 frontier-nofit", True,
           f"all frontier entries exceed the {GB10_USABLE_GB:g} GB window at every "
           f"precision; {min(worst)}-{max(worst)} GB10s needed for weights")


def a15_inkling_pair_differs_only_by_format(report):
    """The Inkling NVFP4/BF16 pair must share geometry and differ only in weights.

    That pair IS the quantization argument: same model, same KV, 3.2x the
    weights. If their KV ever diverges, one of the two arch rows was edited in
    isolation and the comparison stops being like-for-like.
    """
    from helpers.tco_engine import kv_bytes_per_token, _weights_and_raw
    a, b = "Inkling-975B-A41B", "Inkling-975B-A41B (BF16)"
    ka, kb = kv_bytes_per_token(a, "FP8"), kv_bytes_per_token(b, "FP8")
    if ka != kb:
        raise Fail(f"Inkling pair KV diverged: {ka} vs {kb} bytes/token — same "
                   f"architecture must give the same KV")
    wa = _weights_and_raw(a, "FP4")[0]
    wb = _weights_and_raw(b, "BF16")[0]
    ratio = wb / wa
    if not 3.0 <= ratio <= 3.4:
        raise Fail(f"Inkling BF16/NVFP4 weight ratio {ratio:.2f}, expected ~3.22 "
                   f"(592.0 GB vs 1904.8 GB measured from the Hub manifests)")
    report("15 inkling-pair", True,
           f"identical KV ({ka/1024:.1f} KiB/tok), weights {wa:.0f} GB vs {wb:.0f} GB "
           f"({ratio:.2f}x) — format alone")


def a16_predictor_backtest(report):
    """Leave-one-out back-test: the predictor must retrodict its own calibration.

    Calibrate on every measurement but one, predict the held-out one, and
    require the largest relative error to stay under the published ceiling. This
    is the gate that lets a PREDICTED figure publish at all, and it is
    falsifiable: a calibration row that makes the model worse turns this red.

    The published band is this error, NOT the in-sample efficiency spread. A
    min-max interval over n-1 points structurally cannot cover a held-out
    endpoint, so banding on the spread fails by construction and says nothing
    about accuracy.
    """
    from helpers.vllm_reference import loo_errors, _calibration, _PRED_MAX_LOO_ERR
    errs = loo_errors()
    if not errs:
        raise Fail("no leave-one-out errors — calibration set missing or too small")
    if len(errs) < 4:
        raise Fail(f"only {len(errs)} calibration points; need >= 4 to back-test")
    worst = max(abs(e) for e in errs)
    if worst > _PRED_MAX_LOO_ERR:
        raise Fail(f"predictor leave-one-out error {worst:.1%} exceeds the "
                   f"{_PRED_MAX_LOO_ERR:.0%} ceiling — predictions must not publish")
    cal = _calibration()
    if cal is None:
        raise Fail("_calibration() returned None")
    _, med, band, n = cal
    if abs(band - worst) > 1e-9:
        raise Fail(f"published band {band:.3f} != worst LOO error {worst:.3f}; the "
                   f"band must BE the measured out-of-sample error")
    if not 0.3 <= med <= 0.9:
        raise Fail(f"calibrated efficiency {med:.3f} is outside a physical range")
    report("16 predictor-backtest", True,
           f"LOO errors {', '.join(f'{e:+.1%}' for e in errs)} — worst {worst:.1%} "
           f"< {_PRED_MAX_LOO_ERR:.0%}; eff {med:.3f} over n={n}")


def a17_predicted_tier_is_distinct(report):
    """A prediction must never be returned as a measurement, and vice versa.

    `anchor_with_prediction` is what the UI colours off, so the tier label is
    load-bearing: mislabel it and a predicted number is painted green as if it
    were captured on this box.
    """
    from helpers.vllm_reference import anchor_with_prediction, predicted_anchor
    # a measured model must stay measured even though it is also predictable
    t, _why, tier, band = anchor_with_prediction("Qwen2.5-7B")
    if tier != "vllm" or not t:
        raise Fail(f"Qwen2.5-7B should be tier 'vllm', got {tier!r}")
    if band is not None:
        raise Fail("a measurement must not carry a prediction band")
    # every frontier entry must be tier 'predicted', with a band
    for name in _G2B_KV_KIB_FP8:
        t, _why, tier, band = anchor_with_prediction(name)
        if tier != "predicted":
            raise Fail(f"{name}: tier {tier!r}, expected 'predicted'")
        if not t or not band or not (band[0] < t < band[1]):
            raise Fail(f"{name}: point {t} not strictly inside band {band}")
        if "PREDICTED" not in _why:
            raise Fail(f"{name}: provenance string does not say PREDICTED")
    # a vision encoder has no decode rate and must be refused outright
    for enc in ("CLIP ViT-L/14", "ResNet-50"):
        if predicted_anchor(enc)[0] is not None:
            raise Fail(f"{enc} is an encoder — it must not get a tok/s prediction")
    report("17 predicted-tier", True,
           "measured stays 'vllm'; 6 frontier entries are 'predicted' with bands; "
           "encoders refused")


def a18_sparsity_beats_size(report):
    """A sparser MoE must out-run a denser one, even when far larger.

    DeepSeek-V4-Pro is 2.9x Nemotron-3-Ultra's parameters and must still be
    predicted FASTER per user, because it reads 49B per token against 55B. If
    this inverts, the predictor has started following total params and the whole
    sparsity argument the catalog makes is wrong.
    """
    from helpers.vllm_reference import predicted_anchor
    big, _w, _b = predicted_anchor("DeepSeek-V4-Pro (1.6T)")
    mid, _w, _b = predicted_anchor("Nemotron-3-Ultra-550B-A55B")
    if not (big and mid):
        raise Fail("missing predictions for the sparsity comparison")
    if big <= mid:
        raise Fail(f"DeepSeek-V4-Pro (1.6T/49B) predicted {big:.1f} tok/s <= "
                   f"Nemotron-3-Ultra (550B/55B) {mid:.1f} — sparsity term lost")
    # and the largest model must be the slowest per user
    kimi, _w, _b = predicted_anchor("Kimi-K3 (2.8T)")
    if kimi >= mid:
        raise Fail(f"Kimi-K3 (104B active) predicted {kimi:.1f} >= "
                   f"Nemotron-3-Ultra (55B active) {mid:.1f}")
    report("18 sparsity-ranking", True,
           f"1.6T/A49B {big:.1f} tok/s > 550B/A55B {mid:.1f} > 2.8T/A104B {kimi:.1f} — "
           f"ranking follows active params, not total")


def a19_projection_matches_published(report):
    """The cross-hardware projection vs THIRD-PARTY published measurements.

    This is the only external check in the suite, and it covers the one thing
    this box can never measure for itself: what a frontier MoE actually does on
    rack-scale hardware. SemiAnalysis publishes a DeepSeek-R1 671B-A37B FP4
    Pareto curve on GB300 NVL72 at 93 / 162 / 231 tok/s/user. DeepSeek-R1 is the
    closest published analogue to the six 2026 frontier entries -- same sparse
    MoE shape, same MLA attention.

    We project at n_users=1, which is MAXIMUM interactivity, so the right target
    is the TOP of the published envelope, not its middle. Requiring the midpoint
    would be requiring the wrong answer.
    """
    from helpers.vllm_reference import (predicted_on_platform, validation_envelope,
                                        NVL72_REFERENCE)
    env = validation_envelope()
    if not env:
        raise Fail("no cross_hardware_validation block in predicted_anchors.json")
    lo, hi = env
    got = predicted_on_platform("DeepSeek-R1 (671B)", NVL72_REFERENCE, "FP4")
    if not got:
        raise Fail("no projection for the validation model on " + NVL72_REFERENCE)
    tps = got["tps_per_user"]
    # Must not fall below the published low end, and must land within 25% of the
    # high end (n_users=1 == max interactivity == the top of the curve).
    if tps < lo:
        raise Fail(f"projection {tps:.1f} tok/s/user is BELOW the published "
                   f"envelope {lo:.0f}-{hi:.0f} — the projection under-claims")
    if abs(tps - hi) / hi > 0.25:
        raise Fail(f"projection {tps:.1f} tok/s/user is >25% from the published "
                   f"max-interactivity point {hi:.0f} (envelope {lo:.0f}-{hi:.0f})")
    report("19 published-validation", True,
           f"DeepSeek-R1 FP4 on GB300 NVL72 projects {tps:.1f} tok/s/user (TP"
           f"{got['gpus_per_copy']}) vs published {lo:.0f}-{hi:.0f}, "
           f"{abs(tps-hi)/hi:.1%} from the max-interactivity point")


def a20_gb10_basis_is_pass_through(report):
    """The GB10 anchor must be a pure pass-through, not a load-bearing input.

    The UI tells the user that a non-fitting model's GB10 figure is a scaling
    basis rather than a rate, and that 273 GB/s cancels out of the projection.
    That claim is only true if tps_per_user is exactly linear in the basis. If
    some later change makes the basis enter non-linearly, the UI starts lying,
    so this pins the algebra rather than trusting the comment.
    """
    from helpers import tco_engine as T
    from helpers.vllm_reference import NVL72_REFERENCE
    m, prec = "Kimi-K3 (2.8T)", "FP4"
    ratios = []
    for basis in (2.0, 7.0, 100.0):
        r = T.calculate_tco(NVL72_REFERENCE, m, prec, 1, basis, context_len=4096)
        if r.tps_per_user <= 0:
            raise Fail(f"no projection at basis {basis}")
        ratios.append(r.tps_per_user / basis)
    spread = (max(ratios) - min(ratios)) / max(ratios)
    if spread > 1e-9:
        raise Fail(f"tps_per_user is NOT linear in the GB10 basis "
                   f"(ratios {ratios}) — the UI's 'scaling basis, not a rate' "
                   f"explanation is no longer true")
    report("20 basis-pass-through", True,
           f"tps_per_user is exactly linear in the GB10 basis "
           f"(ratio {ratios[0]:.4f} across a 50x basis sweep) — 273 GB/s cancels, "
           f"so the counterfactual GB10 figure cannot distort a projection")


def a21_frontier_has_realistic_platform(report):
    """Every frontier entry must get a real figure on hardware that can hold it.

    The point of the platform-aware predictor: a model that cannot fit a GB10
    must still answer "how fast is it, really?" with a number from a platform
    that can run it -- and that number must be far above the GB10
    counterfactual, since real parts have 8-22 TB/s against the GB10's 273 GB/s.
    """
    from helpers.vllm_reference import (predicted_anchor, smallest_fitting_platform,
                                        predicted_on_platform, NVL72_REFERENCE)
    rows = []
    for m in _G2B_KV_KIB_FP8:
        basis, _w, _b = predicted_anchor(m)
        if not basis:
            raise Fail(f"{m} lost its predicted basis")
        small = smallest_fitting_platform(m)
        rack = predicted_on_platform(m, NVL72_REFERENCE)
        if not small:
            raise Fail(f"{m} has no single-node platform in DEFAULT_SYSTEMS")
        if not rack:
            raise Fail(f"{m} has no projection on {NVL72_REFERENCE}")
        if small["tps_per_user"] <= basis:
            raise Fail(f"{m}: smallest-platform {small['tps_per_user']:.1f} <= "
                       f"GB10 counterfactual {basis:.1f} — scaling term lost")
        # Sanity: a frontier MoE on 2026 rack silicon is tens-to-hundreds of
        # tok/s per user. Outside 10-600 means the projection has broken.
        if not (10.0 <= rack["tps_per_user"] <= 600.0):
            raise Fail(f"{m}: rack projection {rack['tps_per_user']:.1f} tok/s/user "
                       f"is outside the plausible 10-600 band")
        # The GB10 must never be offered as a fitting platform for these.
        if small["system"] == "Dell Pro Max GB10":
            raise Fail(f"{m} reported as fitting a GB10")
        rows.append(f"{m.split()[0]} {rack['tps_per_user']:.0f}")
    report("21 realistic-platform", True,
           "NVL72 tok/s/user: " + ", ".join(rows))


def a22_reproduces_published_curve(report):
    """Reproduce the WHOLE published concurrency curve, not one operating point.

    a19 checks the single-stream ceiling. This checks the shape: at the
    co-residency each published point implies, the engine must land near that
    point's tok/s/user. A single exponent cannot fit three points exactly (the
    pairwise slopes run 0.60-0.82), so the gate is 20%.
    """
    from helpers.vllm_reference import validation_rows, NVL72_REFERENCE
    from helpers.tco_engine import (DELL_SYSTEMS, _CONTENTION_FREE_SESSIONS_PER_GPU,
                                    _SPAN_BW_SHARE_EXP)
    from helpers.vllm_reference import predicted_on_platform
    v = validation_rows()
    if not v:
        raise Fail("no cross_hardware_validation block")
    got = predicted_on_platform("DeepSeek-R1 (671B)", NVL72_REFERENCE, "FP4")
    if not got:
        raise Fail("no projection for the validation model")
    single, tp = got["tps_per_user"], got["gpus_per_copy"]
    knee = _CONTENTION_FREE_SESSIONS_PER_GPU * tp
    worst, rows = 0.0, []
    for pt in v["points"]:
        if pt.get("system") != NVL72_REFERENCE:
            continue
        pub_u = float(pt["tps_per_user"])
        # sessions per CHIP from the published pair, converted to per COPY
        k_copy = (float(pt["tps_per_chip"]) / pub_u) * tp
        model = single * (min(1.0, (k_copy / knee) ** (_SPAN_BW_SHARE_EXP - 1.0)))
        err = (model - pub_u) / pub_u
        worst = max(worst, abs(err))
        rows.append(f"{k_copy:.0f}/copy {model:.0f}v{pub_u:.0f} ({err:+.0%})")
    if not rows:
        raise Fail("no points for " + NVL72_REFERENCE)
    if worst > 0.20:
        raise Fail(f"curve reproduction off by {worst:.0%} (>20%): " + "; ".join(rows))
    report("22 published-curve", True,
           f"all {len(rows)} published points within {worst:.0%}: " + "; ".join(rows))


def a23_spanning_models_contend(report):
    """REGRESSION TEST for the bug this change fixes.

    Contention keyed off per_gpu_density, which was 0 whenever a copy spanned
    GPUs, so every tensor-parallel model reported a FLAT per-user rate at any
    load. Captured failing before the fix: Llama-3.1-405B sat at 165.8 tok/s
    from n=1 to n=256 while single-GPU Qwen2.5-7B correctly decayed 230 -> 132.
    """
    from helpers.tco_engine import calculate_tco
    sysname = "Dell PowerEdge XE9680 (H200 SXM5 (x8) AC)"
    out = []
    for m, prec in (("Llama-3.1-405B", "BF16"), ("Llama-3.3-70B", "BF16"),
                    ("Qwen2.5-7B", "BF16")):
        lo = calculate_tco(sysname, m, prec, 1, 13.1, context_len=4096)
        hi = calculate_tco(sysname, m, prec, 4096, 13.1, context_len=4096)
        if hi.tps_per_user >= lo.tps_per_user:
            raise Fail(f"{m}: per-user tok/s did not fall from n=1 "
                       f"({lo.tps_per_user:.1f}) to n=4096 ({hi.tps_per_user:.1f}) "
                       f"— contention is being skipped (TP{lo.gpus_per_copy})")
        out.append(f"{m} TP{lo.gpus_per_copy} {lo.tps_per_user:.0f}->{hi.tps_per_user:.0f}")
    report("23 spanning-contention", True,
           "per-user decays under load for spanning AND single-GPU: " + ", ".join(out))


def a24_kv_memory_is_provisioned(report):
    """Every concurrent user's KV must be covered by provisioned VRAM.

    The capacity chain is group usable VRAM - weights, / KV per session. This
    asserts the end-to-end guarantee rather than the formula: across models,
    systems and loads, provisioned VRAM >= copies x weights + n_users x KV, and
    the fleet is never packed past the serving reserve.
    """
    from helpers.tco_engine import (calculate_tco, DELL_SYSTEMS, session_footprint,
                                    _SERVING_MEM_FRACTION)
    checked = 0
    for m, prec in (("Inkling-975B-A41B", "FP4"), ("Kimi-K3 (2.8T)", "FP4"),
                    ("Qwen2.5-7B", "BF16"), ("Mixtral-8x7B", "BF16")):
        for sysname in ("Dell PowerEdge XE9780 (B300 AC (x8))",
                        "Dell PowerEdge XE9712 (GB300 NVL72)"):
            for n in (1, 512, 20000):
                r = calculate_tco(sysname, m, prec, n, 20.0, context_len=4096)
                if r.num_nodes <= 0:
                    continue
                w, kv = session_footprint(m, prec, context_len=4096 + 256)
                usable = r.num_nodes * DELL_SYSTEMS[sysname]["vram_gb"] * _SERVING_MEM_FRACTION
                gpn = max(DELL_SYSTEMS[sysname]["gpus_per_node"], 1)
                copies = max(1, r.num_nodes * (gpn // max(r.gpus_per_copy, 1)))
                need = copies * w + n * kv
                if need > usable + 1e-6:
                    raise Fail(f"{m} on {sysname} at n={n}: needs {need:,.0f} GB "
                               f"(weights x{copies} + {n} KV) but only "
                               f"{usable:,.0f} GB usable — KV over-committed")
                checked += 1
    report("24 kv-provisioned", True,
           f"{checked} model/system/load combinations: provisioned VRAM covers "
           f"weights + every concurrent user's KV, inside the "
           f"{_SERVING_MEM_FRACTION:.0%} serving reserve")


def a25_routing_is_diagnostic_only(report):
    """routing_exposure() must inform, never silently change a throughput number.

    The expert-imbalance cost lives in the measured concurrency exponent. If a
    future change turns this diagnostic into a multiplier, the same imbalance
    gets charged twice -- so this pins it as read-only.
    """
    from helpers.tco_engine import routing_exposure, calculate_tco
    if routing_exposure("Llama-3.3-70B", 2) is not None:
        raise Fail("dense model returned a routing exposure")
    lo = routing_exposure("Kimi-K3 (2.8T)", 8, batch_tokens=1)
    hi = routing_exposure("Kimi-K3 (2.8T)", 8, batch_tokens=256)
    if not lo or not hi:
        raise Fail("no routing exposure for Kimi-K3")
    if not lo["exposed"] or hi["exposed"]:
        raise Fail(f"exposure flag wrong: batch1 {lo['exposed']}, batch256 {hi['exposed']}")
    if not (lo["imbalance_cv"] > hi["imbalance_cv"]):
        raise Fail("imbalance spread must shrink as batch grows")
    # read-only: calling it must not perturb the engine
    a = calculate_tco("Dell PowerEdge XE9712 (GB300 NVL72)", "Kimi-K3 (2.8T)", "FP4",
                      256, 20.0, context_len=4096)
    routing_exposure("Kimi-K3 (2.8T)", 8, batch_tokens=256)
    b = calculate_tco("Dell PowerEdge XE9712 (GB300 NVL72)", "Kimi-K3 (2.8T)", "FP4",
                      256, 20.0, context_len=4096)
    if abs(a.tps_per_user - b.tps_per_user) > 1e-9:
        raise Fail("routing_exposure() mutated engine output")
    report("25 routing-diagnostic", True,
           f"dense->None; Kimi-K3 TP8 spread ±{lo['imbalance_cv']:.0%} at batch 1 "
           f"-> ±{hi['imbalance_cv']:.0%} at batch 256; diagnostic does not alter tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="(re)write golden baseline")
    args = ap.parse_args()

    if args.generate:
        generate_baseline()
        return 0

    results = []
    def report(name, ok, msg):
        results.append((name, ok, msg))

    passed = failed = skipped = 0
    for fn in (a1_backcompat, a2_ordering, a3_cost, a4_monotonic, a5_captures, a6_edges,
               a7_toggle, a8_kv_expert_independent, a9_kv_linear_in_context,
               a10_capacity_uses_total, a11_mla_cheaper_than_gqa,
               a12_kv_matches_box_measurements, a13_frontier_kv_geometry,
               a14_frontier_cannot_fit_gb10, a15_inkling_pair_differs_only_by_format,
               a16_predictor_backtest, a17_predicted_tier_is_distinct,
               a18_sparsity_beats_size, a19_projection_matches_published,
               a20_gb10_basis_is_pass_through,
               a21_frontier_has_realistic_platform,
               a22_reproduces_published_curve, a23_spanning_models_contend,
               a24_kv_memory_is_provisioned,
               a25_routing_is_diagnostic_only):
        try:
            fn(report)
        except Fail as e:
            report(fn.__name__, False, str(e))

    print(f"\nTCO capacity smoke  (capacity_mode API: {'present' if HAS_MODE else 'absent'})")
    print("=" * 64)
    for name, ok, msg in results:
        tag = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"  [{tag}] {name}: {msg}")
        passed += ok is True
        failed += ok is False
        skipped += ok is None
    print("=" * 64)
    print(f"  {passed} passed · {failed} failed · {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
