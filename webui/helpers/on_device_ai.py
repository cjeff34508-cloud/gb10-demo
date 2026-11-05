"""
On-device AI narration — runs a local instruct model ON THE Dell Pro Max GB10 to generate
benchmark talking points, TCO recommendations, cloud comparisons, and model
deep-dives. No cloud, no API key. This replaces the earlier Claude-based path.

The narrator is kept RESIDENT on the Dell Pro Max GB10 by default (see streamlit_app's
ensure_narrator), so talking points are instant. It is unloaded only while
benchmarks run, then reloaded. All methods stream token-by-token for
st.write_stream().
"""

from typing import Iterator, Optional

from .llm_inference import LLMInference, _local_path

# Local instruct model used to narrate. Must be downloaded on disk.
# Base narrator: Llama-3.2-3B-Instruct at BF16 (~6 GB weights, ~8 GB resident
# incl. CUDA context) — still trivially light on the 128 GB unified pool, loads
# in a few seconds, no bitsandbytes/OOM quirks, and meaningfully better narration
# quality than the 1.1B TinyLlama it replaces. Its 128K context easily covers our
# prompts, so the modest max_new_tokens budgets below are comfortable.
# (Thinner fallback if needed: TinyLlama/TinyLlama-1.1B-Chat-v1.0 @ FP4.)
# NOTE: uses the ungated unsloth re-host of the weights — the official
# meta-llama repo is gated and this HF account lacks access. Same BF16 weights.
NARRATOR_MODEL = "unsloth/Llama-3.2-3B-Instruct"
NARRATOR_PRECISION = "BF16"


def narrator_available() -> tuple[bool, str]:
    """(usable, reason). The narrator needs its model present on the NVMe."""
    if _local_path(NARRATOR_MODEL) == NARRATOR_MODEL:
        return False, (
            f"On-device narrator model ({NARRATOR_MODEL.split('/')[-1]}) isn't on disk — "
            "download it to enable on-device AI analysis."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Formatting helpers (metrics / TCO rows → compact text for the prompt)
# ---------------------------------------------------------------------------

def _fmt_metrics(rdata: dict) -> str:
    lines = []
    for m in rdata.get("metrics", []):
        if getattr(m, "error", None):
            continue
        bo = getattr(m, "business_output", {}) or {}
        oc = getattr(m, "operational_condition", {}) or {}
        parts = [f"precision={getattr(m, 'precision', '?')}"]
        tps = getattr(m, "tokens_per_sec", 0) or bo.get("tokens_per_sec", 0)
        if tps:
            parts.append(f"{tps:.0f} tok/s")
        if bo.get("ttft_ms"):
            parts.append(f"TTFT {bo['ttft_ms']:.0f} ms")
        if bo.get("images_per_sec"):
            parts.append(f"{bo['images_per_sec']:.0f} img/s")
        if bo.get("tflops"):
            parts.append(f"{bo['tflops']:.0f} TFLOPS")
        if bo.get("bandwidth_gbs"):
            parts.append(f"{bo['bandwidth_gbs']:.0f} GB/s")
        if bo.get("options_per_sec_M"):
            parts.append(f"{bo['options_per_sec_M']:.1f} M options/s")
        if bo.get("paths_per_sec_M"):
            parts.append(f"{bo['paths_per_sec_M']:.1f} M paths/s")
        if getattr(m, "latency_ms", 0):
            parts.append(f"latency {m.latency_ms:.1f} ms")
        if oc.get("mem_pct"):
            parts.append(f"{oc['mem_pct']:.0f}% VRAM")
        parts.append(f"bottleneck={getattr(m, 'primary_bottleneck', '?')}")
        lines.append("  - " + " · ".join(parts))
    return "\n".join(lines) if lines else "  (no successful runs)"


def _fmt_api_costs(api_costs: list[dict]) -> str:
    lines = []
    for c in api_costs:
        lines.append(
            f"  - {c['model']} ({c['provider']}): "
            f"${c['cost_per_request']:.5f}/request, "
            f"${c['cost_per_mtok_out']:,.2f} per 1M output tokens "
            f"(list ${c['in_price']:.2f} in / ${c['out_price']:.2f} out per 1M)"
        )
    return "\n".join(lines) if lines else "  (no API pricing available)"


def _fmt_systems(systems: list[dict]) -> str:
    lines = []
    for s in systems:
        if s.get("mc"):   # FinTech Monte-Carlo sizing
            lines.append(
                f"  - {s['name']}: rating={s['rating']}, "
                f"{s.get('paths_per_sec', 0)/1e9:,.2f}B paths/s, "
                f"mem BW {s['mem_bw_tbs']:.2f} TB/s/GPU, "
                f"{s['nodes']:,} nodes / {s['gpus']:,} GPUs, "
                f"{s['amort']}yr TCO {s['tco']}, {s.get('cost_per_bpaths', 'n/a')}/B-paths"
            )
        else:             # LLM inference
            lines.append(
                f"  - {s['name']}: rating={s['rating']}, "
                f"{s['decode_per_user']:,.0f} tok/s/user, "
                f"cluster {s['cluster_tps']:,.0f} tok/s, "
                f"mem BW {s['mem_bw_tbs']:.2f} TB/s/GPU ({s['link']}), "
                f"{s['nodes']:,} nodes / {s['gpus']:,} GPUs, "
                f"{s['amort']}yr TCO {s['tco']}, {s['cost_per_mtok']}/MTok"
            )
    return "\n".join(lines) if lines else "  (no viable systems)"


def _fmt_workload(w: dict) -> str:
    """Render either an LLM or a FinTech Monte-Carlo workload as one line."""
    if "model" in w:
        return (f"{w['model']} @ {w['precision']}, {w['users']:,} users, "
                f"{w['output_toks']:,} output tokens, {w['context']:,} context, "
                f"{w['amort_years']}yr, ${w['power']:.2f}/kWh")
    return (f"Monte-Carlo bandwidth-bound quant: {w['resident_paths']:,} resident paths × "
            f"{w['bytes_per_path']} B/path, {w['timesteps']} timesteps, target "
            f"{w['target_paths_per_sec']:,} paths/s, {w['amort_years']}yr, ${w['power']:.2f}/kWh")


# ---------------------------------------------------------------------------
# Narrator
# ---------------------------------------------------------------------------

class OnDeviceNarrator:
    """Loads the narrator model on the Dell Pro Max GB10, streams analysis, then unloads."""

    def __init__(self, model: str = NARRATOR_MODEL, precision: str = NARRATOR_PRECISION,
                 inf: Optional[LLMInference] = None):
        """Narrate with an EXISTING loaded model when one is given, else load our own.

        Passing `inf` (a model the Benchmark tab just finished with) means talking
        points and the cloud comparison reuse whatever is already resident instead
        of swapping in a second model. On a 121 GB unified pool that is the whole
        point: a 35B at FP8 plus a separate 3B narrator is two models' worth of
        memory and a load/unload cycle, for prose either one can write.
        """
        self.reason = ""
        self.inf: Optional[LLMInference] = None
        self.owns_inf = False          # only unload what we loaded ourselves
        self.adopted = False

        if inf is not None and getattr(inf, "model", None) is not None:
            self.inf = inf
            self.ready = True
            self.adopted = True
            return

        ok, reason = narrator_available()
        if not ok:
            self.reason, self.ready = reason, False
            return
        self.inf = LLMInference(model, precision)
        self.owns_inf = True
        self.ready = bool(self.inf.load_model())
        if not self.ready:
            self.reason = "On-device narrator failed to load."

    @property
    def model_label(self) -> str:
        """Which model is actually writing the prose, for the UI to show."""
        if self.inf is None:
            return "none"
        name = getattr(self.inf, "model_name", "?").split("/")[-1]
        prec = getattr(self.inf, "_effective_precision", None) or getattr(self.inf, "precision", "")
        return f"{name} @ {prec}" + (" (reused from benchmark)" if self.adopted else "")

    def unload(self):
        """Release the model ONLY if this narrator loaded it.

        An adopted instance belongs to the Benchmark tab, which is responsible for
        its lifetime — unloading it here would silently free a model the caller
        still has a handle to.
        """
        if self.inf is not None and self.owns_inf:
            self.inf.unload()
        self.inf = None
        self.ready = False

    def _stream(self, system: str, user: str, max_new_tokens: int = 384) -> Iterator[str]:
        if not self.ready or self.inf is None:
            yield f"⚠ {self.reason}"
            return
        yield from self.inf.generate_stream(user, system=system, max_new_tokens=max_new_tokens)

    # -- results talking points -------------------------------------------------
    def results_summary(self, rdata: dict) -> Iterator[str]:
        system = (
            "You are a Dell + NVIDIA solutions engineer presenting Dell Pro Max GB10 (Grace-Blackwell, "
            "128 GB unified LPDDR5X at ~273 GB/s) benchmark results to a technical buyer. "
            "Use ONLY the numbers provided — never invent benchmarks. Be concise and "
            "vendor-credible. Markdown bullets, no preamble."
        )
        facts = _fmt_metrics(rdata)
        # Refuse to narrate without numbers. _fmt_metrics reads BenchmarkMetrics
        # OBJECTS via getattr; handed anything else it degrades to
        # "precision=? · bottleneck=?", and a model told to "use ONLY the numbers
        # provided" will cheerfully invent a throughput figure instead of saying
        # it has none. An empty talking-points panel is recoverable; a fabricated
        # benchmark number in front of a customer is not.
        if not facts.strip() or "precision=?" in facts:
            return iter(["⚠ No usable metrics to summarise — run a benchmark first. "
                         "(Talking points are generated only from measured results.)"])
        user = (
            f"Scenario: {rdata.get('scenario')}\nModel/Test: {rdata.get('model')}\n"
            f"Per-precision results:\n{facts}\n\n"
            "Write 4 short bullet talking points (headline result, what the bottleneck means, "
            "the effect of precision, one honest caveat), then one bold one-line takeaway."
        )
        return self._stream(system, user, max_new_tokens=320)

    # -- TCO recommendation -----------------------------------------------------
    def tco_recommendation(self, workload: dict, systems: list[dict]) -> Iterator[str]:
        system = (
            "You are a Dell + NVIDIA TCO advisor writing a customer-facing hardware "
            "recommendation. Performance is driven by per-GPU memory bandwidth (HBM3 >> HBM2) "
            "and the interconnect used to span a model (NVLink >> PCIe >> cross-node "
            "InfiniBand); VRAM sets fit and sessions-per-node. Use ONLY the provided figures. "
            "Markdown, no preamble."
        )
        user = (
            f"Workload: {_fmt_workload(workload)}.\n\n"
            f"Candidate systems (ranked best→worst):\n{_fmt_systems(systems)}\n\n"
            "Recommend: (1) the best fit and WHY (memory bandwidth / interconnect / VRAM / "
            "cost-efficiency), (2) a runner-up and its trade-off, (3) any trap to avoid (forced "
            "multi-node, or a low-bandwidth part needing far more GPUs), then (4) one bold "
            "one-line recommendation."
        )
        return self._stream(system, user, max_new_tokens=400)

    # -- cloud comparison (on-prem Dell Pro Max GB10 vs hosted Claude/Gemini APIs) ------------
    def cloud_comparison(self, workload: dict, onprem: Optional[dict],
                         api_costs: Optional[list[dict]] = None,
                         band: Optional[dict] = None) -> Iterator[str]:
        system = (
            "You are a Dell + NVIDIA solutions engineer comparing running an LLM workload "
            "ON-PREM on the customer's BEST-FIT Dell platform (named below, with its amortized "
            "$/1M output tokens) versus paying per-token for a hosted frontier API (Anthropic "
            "Claude / Google Gemini). The per-token figures provided are published list prices; "
            "the on-prem cost/MTok is amortized hardware+power over the chosen term. Compare on "
            "the SAME basis ($ per 1M output tokens). Be balanced — note where the API genuinely "
            "wins (zero CapEx, elastic burst, frontier quality, no ops). Label derived figures as "
            "estimates and never invent prices. Markdown bullets, no preamble."
        )
        if onprem:
            otxt = (f"Best on-prem option — {onprem.get('name', 'Dell platform')}: "
                    f"{onprem.get('decode_per_user', 0):,.0f} tok/s/user, amortized "
                    f"{onprem.get('cost_per_mtok', 'n/a')} per 1M output tokens, "
                    f"{onprem.get('tco', 'n/a')} {workload.get('amort_years', '?')}yr TCO, "
                    f"fits={onprem.get('fits', 'n/a')}.")
        else:
            otxt = "No selected Dell platform can host this exact model/precision."

        # Frame the hosted-API side as a low / price-point / high band when available,
        # otherwise fall back to the full cheapest-first list.
        def _anchor(c: dict) -> str:
            return f"{c['model']}: ${c['cost_per_mtok_out']:,.2f}/1M output tok"
        if band and band.get("point"):
            lines = []
            if band.get("low"):   lines.append(f"Low anchor — {_anchor(band['low'])}")
            lines.append(f"PRICE POINT — {_anchor(band['point'])}")
            if band.get("high"):  lines.append(f"High anchor — {_anchor(band['high'])}")
            api_txt = ("Hosted API cost band for ONE such request:\n" + "\n".join(lines))
            task = (
                "Make the on-prem-vs-API cost case, framed around the hosted-API PRICE POINT: "
                "(1) compare the best on-prem platform's amortized $/1M output tokens against the "
                "PRICE POINT, and bracket it with the low and high anchors — who wins at this "
                "workload and by roughly how much across that range; "
            )
        else:
            api_txt = ("Hosted API token cost for ONE such request (cheapest first):\n"
                       f"{_fmt_api_costs(api_costs or [])}")
            task = (
                "Make the on-prem-vs-API cost case: (1) compare the best on-prem platform's "
                "amortized $/1M output tokens against the cheapest and a frontier API option — "
                "who wins at this workload and by roughly how much; "
            )
        user = (
            f"Workload: {_fmt_workload(workload)}.\n{otxt}\n\n"
            f"{api_txt}\n\n"
            + task +
            "(2) the rough monthly request volume at which the on-prem TCO pays for itself vs "
            "per-token billing (call it an estimate); (3) non-cost advantages of on-prem (data "
            "stays local, no egress, fixed latency, always-on) and when the hosted API is the "
            "better call (bursty/low volume, need frontier quality, zero ops). End with a bold "
            "one-line verdict."
        )
        return self._stream(system, user, max_new_tokens=420)

    # -- model deep-dive --------------------------------------------------------
    def explain_model(self, name: str, scenario: str, curated_body: str = "") -> Iterator[str]:
        system = (
            "You are an ML systems expert explaining a model or HPC workload to a technical "
            "audience evaluating the Dell Pro Max GB10 (Grace-Blackwell, 128 GB unified LPDDR5X "
            "~273 GB/s, Blackwell tensor cores with FP4 acceleration). Be accurate about "
            "architecture and whether it is compute-bound or memory-bandwidth-bound on this "
            "chip. If unsure of an exact number, say so. Markdown, no preamble."
        )
        extra = f"\n\nKnown facts (treat as ground truth):\n{curated_body}" if curated_body else ""
        user = (
            f"Give a detailed deep-dive on '{name}' as used in the Dell Pro Max GB10 scenario '{scenario}'. "
            "Cover: architecture & params, what it's good at, compute-bound vs "
            "memory-bandwidth-bound on the Dell Pro Max GB10 and why, precision behavior "
            "(FP32/FP16/INT8/FP4) and the VRAM/throughput trade-offs, and when a buyer would "
            f"choose it. Keep it tight and concrete.{extra}"
        )
        return self._stream(system, user, max_new_tokens=400)


def make_narrator(inf: Optional[LLMInference] = None) -> OnDeviceNarrator:
    """Narrator over an existing loaded model if `inf` is given, else load one.

    (Call inside a spinner; remember to .unload() — which is a no-op for an
    adopted model, since the Benchmark tab owns that one.)
    """
    return OnDeviceNarrator(inf=inf)
