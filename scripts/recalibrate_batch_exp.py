#!/usr/bin/env python3
"""Fit _BATCH_BW_SHARE_EXP to the captured vLLM anchors.

The shared-weights capacity model scales aggregate throughput as
    aggregate(n) = n * baseline * n**(exp - 1)  ==  baseline * n**exp
so a log-log fit of aggregate vs concurrency gives the exponent directly:
    exp = slope of log(aggregate/baseline) vs log(n)

Reports the fit per capture and the pooled value, and compares with the value
currently pinned in tco_engine.py. Does NOT edit the engine — assertion 5 bounds
the exponent, and a calibration constant should move only with its reasoning
recorded.
"""
import json, math, re
from pathlib import Path

ROOT = Path.home() / "gb10-demo"
ref = json.loads((ROOT / "webui" / "data" / "vllm_reference.json").read_text())
src = (ROOT / "webui" / "helpers" / "tco_engine.py").read_text()
cur = float(re.search(r"_BATCH_BW_SHARE_EXP\s*=\s*([\d.]+)", src).group(1))

def fit(rows):
    """Least-squares slope of log(agg) vs log(n), anchored at n=1."""
    pts = [(r["n_users"], r["aggregate_tps"]) for r in rows
           if r.get("n_users") and r.get("aggregate_tps")]
    if len(pts) < 3:
        return None, len(pts)
    base = next((a for n, a in pts if n == 1), None)
    if not base:
        return None, len(pts)
    xs = [math.log(n) for n, _ in pts if n > 1]
    ys = [math.log(a / base) for n, a in pts if n > 1]
    if len(xs) < 2:
        return None, len(pts)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None), len(pts)

print(f"currently pinned _BATCH_BW_SHARE_EXP = {cur}")
print(f"{'capture':34s} {'prec':6s} {'levels':>6s} {'fitted exp':>11s}")
fits = []
for k, c in (ref.get("captures") or {}).items():
    e, n = fit(c.get("per_n") or [])
    fits.append((k, e)) if e else None
    print(f"  {k:32s} {c.get('precision','?'):6s} {n:6d} "
          f"{(f'{e:.3f}' if e else 'n/a (need >=3 levels incl n=1)'):>11s}")
good = [e for _, e in fits if e]
if good:
    pooled = sum(good) / len(good)
    print(f"\npooled over {len(good)} capture(s): {pooled:.3f}   (pinned {cur})")
    print(f"delta {pooled - cur:+.3f}")
    print("\n1.0 = perfectly linear aggregate scaling; lower = sub-linear.")
    print("Assertion 5 requires the analytic aggregate within [0.5, 2.0]x the")
    print("captures, so a small delta needs no change — only a documented one does.")
else:
    print("\nno capture has >=3 concurrency levels including n=1 — cannot fit")
