"""
Cloud API token-cost reference for the TCO "Compare to Cloud" comparison.

Published per-million-token list prices (USD) for hosted frontier LLM APIs, used
to estimate what the user's workload would cost on Claude / Gemini instead of
running it on-prem on the Dell GB10. These are standard-context paid-tier rates as of
2026 — verify before quoting in a customer-facing setting.

Sources:
  - Claude:  platform.claude.com / the claude-api skill model table
  - Gemini:  ai.google.dev/gemini-api/docs/pricing
"""

# Per-model: provider, input $/1M tokens, output $/1M tokens, one-line note.
# Curated to span premium → budget so the comparison shows a real range.
API_PRICING: dict[str, dict] = {
    # --- Anthropic Claude -------------------------------------------------
    "Claude Opus 4.8":       {"provider": "Anthropic", "in": 5.00, "out": 25.00,
                              "note": "most capable Claude"},
    "Claude Sonnet 4.6":     {"provider": "Anthropic", "in": 3.00, "out": 15.00,
                              "note": "balanced Claude"},
    "Claude Haiku 4.5":      {"provider": "Anthropic", "in": 1.00, "out":  5.00,
                              "note": "fast/cheap Claude"},
    # --- Google Gemini (standard <=200K context) --------------------------
    "Gemini 3.1 Pro":        {"provider": "Google",    "in": 2.00, "out": 12.00,
                              "note": "Gemini flagship"},
    "Gemini 3.5 Flash":      {"provider": "Google",    "in": 1.50, "out":  9.00,
                              "note": "fast Gemini"},
    "Gemini 2.5 Flash":      {"provider": "Google",    "in": 0.30, "out":  2.50,
                              "note": "low-cost Gemini"},
    "Gemini 2.5 Flash-Lite": {"provider": "Google",    "in": 0.10, "out":  0.40,
                              "note": "cheapest Gemini"},
}


def api_token_costs(input_toks: int, output_toks: int) -> list[dict]:
    """Per-model cost of one request that reads ``input_toks`` and writes
    ``output_toks``.

    Each entry also carries ``cost_per_mtok_out`` — the cost per 1M OUTPUT
    tokens, loaded with the proportional input each output token "carries"
    (output_price + input_price × input/output). That figure is on the same
    output-token basis as the Dell GB10 TCO's ``cost_per_mtok``, so the two are
    directly comparable. Sorted cheapest → most expensive by that figure.
    """
    out_ratio = (input_toks / output_toks) if output_toks else 0.0
    rows: list[dict] = []
    for name, p in API_PRICING.items():
        per_request   = input_toks / 1e6 * p["in"] + output_toks / 1e6 * p["out"]
        per_mtok_out  = p["out"] + p["in"] * out_ratio
        rows.append({
            "model":             name,
            "provider":          p["provider"],
            "note":              p["note"],
            "in_price":          p["in"],
            "out_price":         p["out"],
            "cost_per_request":  per_request,
            "cost_per_mtok_out": per_mtok_out,
        })
    rows.sort(key=lambda d: d["cost_per_mtok_out"])
    return rows


def cheapest_api(input_toks: int, output_toks: int) -> dict:
    """The single cheapest API option for this workload (by $/M output tok)."""
    return api_token_costs(input_toks, output_toks)[0]
