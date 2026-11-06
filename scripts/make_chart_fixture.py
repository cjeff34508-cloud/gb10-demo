#!/usr/bin/env python3
"""Draw a deterministic test chart whose values are known exactly.

The vision track's encoder benchmark feeds SYNTHETIC RANDOM pixels and reports
images/sec — a number that is true and says nothing about whether the model
understood the image. A chart with values we chose is checkable: ask the model
to read the tallest bar and compare the answer to the number we drew.

Uses PIL, which every workload venv already has — no new dependency.
Writes webui/assets/fixture-chart.png plus a .truth.json beside it.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Real measurements from this box, so the fixture doubles as a talking point.
# READ from webui/data/vllm_reference.json rather than hardcoded: the figures
# were transcribed once and then drifted (the chart said 36.4 and 61.0 while the
# reference file said 38.8 and 50.8), so a customer-facing chart captioned
# "measured on GB10" disagreed with the number the TCO tab used for the same
# model in the same session. Deriving it means that cannot recur.
_REF = Path(__file__).resolve().parent.parent / "webui" / "data" / "vllm_reference.json"
_LABELS = {                      # capture key -> short bar label
    "qwen2.5-7b":          "Qwen2.5-7B BF16",
    "qwen3-8b-nvfp4":      "Qwen3-8B NVFP4",
    "qwen3.6-35b-a3b-fp8": "Qwen3.6-35B-A3B FP8",
}


def _single_stream(cap: dict) -> float | None:
    """n=1 per-user tok/s from a capture, or None."""
    for r in cap.get("per_n") or []:
        if r.get("n_users") == 1 and r.get("per_user_tps"):
            return float(r["per_user_tps"])
    return None


_caps = (json.loads(_REF.read_text()).get("captures") or {})
DATA = {}
for _k, _label in _LABELS.items():
    _v = _single_stream(_caps.get(_k) or {})
    if _v is not None:
        DATA[_label] = round(_v, 1)
if len(DATA) < 2:
    # Fail loudly. Drawing a chart captioned as a measurement from stale or
    # missing data is the failure this script exists to prevent.
    raise SystemExit(
        f"refusing to draw: only {len(DATA)} of {len(_LABELS)} captures have an "
        f"n=1 level in {_REF}. Run scripts/recapture_reference.sh first."
    )
print("fixture values from vllm_reference.json:", DATA)
TITLE = "Single-stream decode on GB10"
UNITS = "tokens per second"
COLORS = [(137, 135, 129), (42, 120, 214), (0, 131, 0)]

W, H = 760, 480
PAD_L, PAD_B, PAD_T = 90, 90, 70
plot_h = H - PAD_T - PAD_B
ymax = max(DATA.values()) * 1.25

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

def font(sz, bold=False):
    for p in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()

f_title, f_lbl, f_val, f_ax = font(22, True), font(14), font(18, True), font(13)

d.text((PAD_L, 22), TITLE, fill="black", font=f_title)
# axes
d.line([(PAD_L, PAD_T), (PAD_L, H - PAD_B)], fill="black", width=2)
d.line([(PAD_L, H - PAD_B), (W - 30, H - PAD_B)], fill="black", width=2)
# y ticks
for i in range(0, 5):
    v = ymax * i / 4
    y = H - PAD_B - plot_h * (v / ymax)
    d.line([(PAD_L - 6, y), (PAD_L, y)], fill="black", width=2)
    d.text((PAD_L - 48, y - 8), f"{v:.0f}", fill="black", font=f_ax)
d.text((12, PAD_T - 34), UNITS, fill="black", font=f_ax)

n = len(DATA)
slot = (W - 30 - PAD_L) / n
bw = slot * 0.52
for i, ((label, val), col) in enumerate(zip(DATA.items(), COLORS)):
    cx = PAD_L + slot * (i + 0.5)
    bh = plot_h * (val / ymax)
    x0, y0, x1, y1 = cx - bw / 2, H - PAD_B - bh, cx + bw / 2, H - PAD_B
    d.rectangle([x0, y0, x1, y1], fill=col)
    t = f"{val:.1f}"
    tw = d.textlength(t, font=f_val)
    d.text((cx - tw / 2, y0 - 26), t, fill="black", font=f_val)
    for j, part in enumerate(label.split(" ")):
        pw = d.textlength(part, font=f_lbl)
        d.text((cx - pw / 2, H - PAD_B + 8 + j * 17), part, fill="black", font=f_lbl)

out = Path.home() / "gb10-demo" / "webui" / "assets" / "fixture-chart.png"
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out)

truth = {
    "series": DATA, "title": TITLE, "units": UNITS,
    "tallest_label": max(DATA, key=DATA.get),
    "tallest_value": max(DATA.values()),
    "shortest_label": min(DATA, key=DATA.get),
    "shortest_value": min(DATA.values()),
}
out.with_suffix(".truth.json").write_text(json.dumps(truth, indent=1))
print(f"■ wrote {out} ({out.stat().st_size/1024:.0f} KB)")
print(json.dumps(truth, indent=1))
