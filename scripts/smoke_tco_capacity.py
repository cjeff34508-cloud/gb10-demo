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
    for fn in (a1_backcompat, a2_ordering, a3_cost, a4_monotonic, a5_captures, a6_edges, a7_toggle):
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
