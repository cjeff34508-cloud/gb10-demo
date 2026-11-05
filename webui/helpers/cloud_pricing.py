"""
Cloud API token-cost reference for the TCO "Compare to Cloud" comparison.

Published per-million-token list prices (USD) for hosted frontier LLM APIs, used
to estimate what the user's workload would cost on Claude / Gemini instead of
running it on-prem on the Dell Pro Max GB10.

================================================================================
STANCE: this comparison is deliberately CONSERVATIVE TOWARD ON-PREM.
================================================================================
Reviewed 2026-11-05. Where a published price has a RANGE, we take the high end.
Where we have a real measured or published number, we use it unchanged — we do
not inflate prices, and we do not keep a stale rate because it happens to be
higher. Every lever below is real published pricing structure, not a thumb on
the scale:

  1. Long-context tiers are applied.   Several models charge more above a prompt
     threshold (Claude Haiku 5.5 above 100K; Gemini 3.1 Pro above 200K). The TCO
     tab defaults to 131,072 context and goes to 524,288, so the higher tier is
     the NORMAL case here, not an edge case. Ignoring it — as this module did
     before — understated cloud cost at every default setting.
  2. Thinking tokens bill as OUTPUT tokens. Every current model runs adaptive
     thinking by default, so a reasoning workload is billed for more output than
     it returns. See THINKING_OUTPUT_MULTIPLIER — that one is a labelled
     ASSUMPTION, not a measurement, and it is the only estimated input here.
  3. The quoted price point is a frontier-class model, not the cheapest tier.
     A customer asking "what would this cost on an API" usually means at
     comparable quality — and this box's own cloud escalations run Claude Opus.
  4. Cache reads and the 50% Batch discount are NOT applied. Both are real and
     both would lower cloud cost; omitting them is a conservative choice and is
     recorded here so the omission is visible rather than accidental.

The cheap tiers remain in the table and remain the `low` anchor, so the full
honest range is still on screen. Anyone can see where the stance sits.

Sources:
  - Claude:  the claude-api skill model table (cached 2026-10-06), first-party
             Anthropic API rates. Verified for every Claude row below.
  - Gemini:  ai.google.dev/gemini-api/docs/pricing — see VERIFIED flags; the
             Gemini rows could NOT be confirmed against Google's own page on
             2026-11-05 and are marked accordingly.
"""

# Per-model:
#   provider, in/out $ per 1M tokens, note
#   gen      — "current" or "previous" (previous kept so "what about Opus 4.8?" has an answer)
#   verified — True only where the rate was confirmed against a first-party source
#   long     — real higher tier above a prompt-token threshold: {"over", "in", "out"}
#   fast     — real premium-speed rate, where the provider publishes one
API_PRICING: dict[str, dict] = {
    # --- Anthropic Claude, current generation (verified) ------------------
    "Claude Fable 5.1":      {"provider": "Anthropic", "in": 10.00, "out": 50.00,
                              "gen": "current", "verified": True,
                              "note": "most capable widely released Claude"},
    "Claude Opus 5.5":       {"provider": "Anthropic", "in":  4.00, "out": 20.00,
                              "gen": "current", "verified": True,
                              "fast": {"in": 8.00, "out": 40.00},
                              "note": "current Opus; fast mode 2x"},
    "Claude Opus 5":         {"provider": "Anthropic", "in":  5.00, "out": 25.00,
                              "gen": "current", "verified": True,
                              "note": "the model this box escalates to"},
    "Claude Sonnet 5.5":     {"provider": "Anthropic", "in":  2.00, "out": 10.00,
                              "gen": "current", "verified": True,
                              "note": "current Sonnet"},
    "Claude Haiku 5.5":      {"provider": "Anthropic", "in":  0.10, "out":  0.50,
                              "gen": "current", "verified": True,
                              "long": {"over": 100_000, "in": 0.50, "out": 2.50},
                              "note": "cheapest Claude; 5x above 100K prompt"},
    # --- Anthropic Claude, previous generation (verified, kept for Q&A) ---
    "Claude Opus 4.8":       {"provider": "Anthropic", "in":  5.00, "out": 25.00,
                              "gen": "previous", "verified": True,
                              "note": "previous-gen Opus"},
    "Claude Sonnet 4.6":     {"provider": "Anthropic", "in":  3.00, "out": 15.00,
                              "gen": "previous", "verified": True,
                              "note": "previous-gen Sonnet"},
    "Claude Haiku 4.5":      {"provider": "Anthropic", "in":  1.00, "out":  5.00,
                              "gen": "previous", "verified": True,
                              "note": "previous-gen Haiku; 200K context only"},
    # --- Google Gemini — UNVERIFIED, see module docstring -----------------
    "Gemini 3.1 Pro":        {"provider": "Google",    "in":  2.00, "out": 12.00,
                              "gen": "current", "verified": False,
                              "long": {"over": 200_000, "in": 4.00, "out": 18.00},
                              "note": "Gemini flagship; 2x above 200K prompt"},
    "Gemini 3.5 Flash":      {"provider": "Google",    "in":  1.50, "out":  9.00,
                              "gen": "current", "verified": False,
                              "note": "fast Gemini"},
    "Gemini 2.5 Flash":      {"provider": "Google",    "in":  0.30, "out":  2.50,
                              "gen": "previous", "verified": False,
                              "note": "low-cost Gemini"},
    "Gemini 2.5 Flash-Lite": {"provider": "Google",    "in":  0.10, "out":  0.40,
                              "gen": "previous", "verified": False,
                              "note": "cheapest Gemini"},
}

# The ONE estimated input in this module. Thinking tokens are billed at output
# rates on every current model, and adaptive thinking is on by default, so a
# reasoning workload is billed for more output than the user receives. We have
# no measured ratio for the customer's workload, so this is an explicit,
# adjustable assumption rather than a hidden fudge — set to 1.0 for a
# thinking-free workload (classification, extraction, short Q&A).
# 1.5 = "billed output is 50% above returned output", a deliberately
# conservative-toward-on-prem reading for a reasoning/agentic workload.
THINKING_OUTPUT_MULTIPLIER: float = 1.5

# What the comparison quotes. Deliberately frontier-weighted (see STANCE #3);
# `low` keeps the genuinely cheap option visible so the range stays honest.
# Names MUST match API_PRICING keys above.
CLOUD_ANCHORS: dict[str, str] = {
    "low":   "Claude Haiku 5.5",   # cheapest current Claude — the honest floor
    "point": "Claude Opus 5.5",    # the rate the comparison quotes (current Opus)
    "high":  "Claude Fable 5.1",   # real frontier ceiling
}

# One-line provenance string the UI renders next to the comparison, so the
# stance is never implicit.
STANCE_LABEL = (
    "Conservative toward on-prem: long-context tiers applied, thinking tokens "
    f"billed at {THINKING_OUTPUT_MULTIPLIER:g}x returned output, frontier-class "
    "price point, and cache/batch discounts deliberately NOT applied. "
    "Claude rates verified 2026-11-05; Gemini rates unverified."
)


def effective_rates(name: str, input_toks: int,
                    fast: bool = False) -> tuple[float, float, str | None]:
    """``(in_price, out_price, tier_note)`` for this prompt size.

    Applies the model's real long-context tier when ``input_toks`` exceeds its
    published threshold, and the published premium-speed rate when ``fast``.
    """
    p = API_PRICING[name]
    in_p, out_p, tier = p["in"], p["out"], None
    long = p.get("long")
    if long and input_toks > long["over"]:
        in_p, out_p = long["in"], long["out"]
        tier = f">{long['over'] // 1000}K prompt tier"
    if fast and p.get("fast"):
        in_p, out_p = p["fast"]["in"], p["fast"]["out"]
        tier = "fast mode" if not tier else f"{tier} + fast mode"
    return in_p, out_p, tier


def api_token_costs(input_toks: int, output_toks: int, *,
                    thinking_multiplier: float | None = None,
                    include_previous_gen: bool = True,
                    fast: bool = False) -> list[dict]:
    """Per-model cost of one request reading ``input_toks`` / writing ``output_toks``.

    ``cost_per_mtok_out`` is the cost per 1M OUTPUT tokens, loaded with the
    proportional input each output token "carries" (out + in x input/output), on
    the same output-token basis as the GB10 TCO's ``cost_per_mtok`` — so the two
    are directly comparable. Sorted cheapest -> most expensive by that figure.

    Billed output = returned output x ``thinking_multiplier`` (see
    THINKING_OUTPUT_MULTIPLIER). Long-context tiers are applied automatically.
    """
    mult = THINKING_OUTPUT_MULTIPLIER if thinking_multiplier is None else thinking_multiplier
    billed_out = output_toks * mult
    out_ratio = (input_toks / billed_out) if billed_out else 0.0
    rows: list[dict] = []
    for name, p in API_PRICING.items():
        if not include_previous_gen and p.get("gen") == "previous":
            continue
        in_p, out_p, tier = effective_rates(name, input_toks, fast=fast)
        rows.append({
            "model":             name,
            "provider":          p["provider"],
            "note":              p["note"],
            "gen":               p.get("gen", "current"),
            "verified":          bool(p.get("verified")),
            "in_price":          in_p,
            "out_price":         out_p,
            "tier":              tier,
            "thinking_mult":     mult,
            "billed_out_toks":   billed_out,
            "cost_per_request":  input_toks / 1e6 * in_p + billed_out / 1e6 * out_p,
            "cost_per_mtok_out": out_p + in_p * out_ratio,
        })
    rows.sort(key=lambda d: d["cost_per_mtok_out"])
    return rows


def api_comparison_band(input_toks: int, output_toks: int, **kw) -> dict[str, dict]:
    """The low / price-point / high anchors for the on-prem-vs-cloud comparison.

    Returns ``{"low": row, "point": row, "high": row}``, each a full cost dict
    from :func:`api_token_costs`. A missing anchor is simply omitted.
    """
    by_name = {c["model"]: c for c in api_token_costs(input_toks, output_toks, **kw)}
    return {role: by_name[name] for role, name in CLOUD_ANCHORS.items() if name in by_name}


def cheapest_api(input_toks: int, output_toks: int, **kw) -> dict:
    """The single cheapest API option for this workload (by $/M output tok)."""
    return api_token_costs(input_toks, output_toks, **kw)[0]
