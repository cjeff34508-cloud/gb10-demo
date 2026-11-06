# Contention, spanning, KV provisioning and routing — 2026-11-06

Why 43 of the 90 golden-baseline rows moved, row group by row group, and what was
deliberately left alone. The pre-change baseline is kept at
`scripts/tco_golden_baseline.pre-contention.json`.

## The bug that started it

`tco_engine.py` computed bandwidth contention from `per_gpu_density`, which was set to
`0` whenever one model copy spanned more than one GPU. Contention keyed off that density,
so **every tensor-parallel model reported a flat per-user rate at any load**:

| model | TP | n=1 | n=256 |
|---|---|---|---|
| `Qwen2.5-7B` | 1 | 230.3 tok/s | 132.3 — correct decay |
| `Llama-3.3-70B` | 2 | 216.5 | **216.5 — unchanged** |
| `Llama-3.1-405B` | 8 | 165.8 | **165.8 — unchanged** |

All six 2026 frontier entries span GPUs, so all six were affected. It also contradicted
the demo script's own credibility guardrail: *"sessions/node is a MEMORY figure — always
pair it with per-user tok/s."* Now pinned by assertion **a23**, which is a real regression
test: it fails on the old code.

## A second bug, found while fixing the first

Co-residency was `min(per_gpu_density, n_users)` — the *fully packed* density against the
*whole fleet's* user count. Both halves were wrong:

- it charged a copy for every user in the deployment, when a 72-GPU node running a TP8
  model holds **9 independent copies** and 1,000 users is 112 sessions each, not 1,000;
- it assumed every user lands on one device, so 8 users spread across 8 GPUs were billed
  as 8 co-resident on one.

Co-residency is now the load on the busiest copy: `ceil(n_users / total_copies)`, capped
at what one copy can hold. Old and new converge at full packing (0.333 vs 0.354 for
`Qwen2.5-7B` at n=64) and diverge at partial load, which is where the old model was wrong.

## Calibrating the spanning exponent on published data, not a guess

This box cannot measure multi-GPU anything, so the spanning exponent comes from
third-party data: SemiAnalysis InferenceX's DeepSeek-R1 671B-A37B FP4 curve on GB300
NVL72, `(tok/s/user, tok/s/chip)` = (231, 1065.2), (162, 5393.8), (93, 12534.4).
Dividing gives co-resident sessions per chip of 4.61 / 33.3 / 134.8, and a least-squares
fit of `per-user ~ k**(exp-1)` gives **`_SPAN_BW_SHARE_EXP = 0.7364`**.

It is deliberately steeper than the **0.90** measured on this box, and that gap is the
constant's content: a spanning MoE copy loses more to concurrency than a single GB10,
because at high batch it also pays all-to-all expert dispatch/combine, expert-routing
imbalance and compute saturation.

### The knee, and the mistake that nearly shipped without it

Applying the power law from `k=1` over-charges contention at low concurrency: it predicted
**130 tok/s/user where the published curve measures 231**. Solving for where the power law
meets the single-stream ceiling gives a **contention-free region of 5.36 sessions/GPU**,
and both datasets support one independently — this box's vLLM capture has per-user tok/s
*rising* 13.1 → 14.4 from 1 to 32 users.

With the knee, **all three published points reproduce within 11%** (+1.3%, −10.8%, +7.5%)
— validation across the whole curve rather than at one operating point (assertion **a22**).

**The knee applies only to shared weights.** It exists because a batch amortises *one*
weight read; replicated copies share nothing and each re-streams its own weights, so they
contend from the second copy with no knee. Applying it to replicated mode inflated
baseline per-user tok/s by ~2.4×, which is how the error was caught.

## Spanning is no longer a special case that drops the model

A copy spanning nodes used to collapse to **one session per copy**, which is why Kimi-K3
on an XE9680 read 196 nodes and $512/MTok. Multi-node copies batch in production — that is
how 671B+ models are served — so they now get the same capacity model and pay for the span
through `inter_eff` instead.

## KV for concurrent users: it was modelled, and it is now provisioned honestly

The capacity chain always was *group usable VRAM − weights, ÷ KV per session*, and an audit
confirmed it never over-commits. But it sized against **100% of nameplate VRAM**, packing
an Inkling fleet to 99.5% of spec — which no server achieves. vLLM's documented default is
`gpu_memory_utilization=0.9`, the remainder going to activations, CUDA-graph capture, the
framework and allocator fragmentation.

`_SERVING_MEM_FRACTION = 0.90` now applies to every system. This is the single largest
source of baseline movement: capacity drops ~10–11% across the board, in the honest
direction. Assertion **a24** pins the end-to-end guarantee across 24 model/system/load
combinations.

A model whose weights leave no room for even one session's KV now returns **"Cannot fit"**
rather than claiming one session via `max(1, ...)` and printing a $/MTok.

## Routing and load balancing — named, deliberately not costed

`MODEL_ARCH` has carried `experts` and `topk` since the catalog was written and nothing
ever computed with them. `routing_exposure()` is their first use, and it returns a
**diagnostic, never a throughput multiplier**:

| model | TP | batch | activations/GPU/step | uniform-routing spread |
|---|---|---|---|---|
| `Kimi-K3` (896 experts, top-16) | 8 | 1 | 2.0 | **±66%** |
| `Kimi-K3` | 8 | 256 | 512 | ±4% |
| `DeepSeek-V4-Pro` (384, top-6) | 4 | 1 | 1.5 | **±71%** |
| `Llama-3.3-70B` (dense) | 2 | — | — | `None` |

The restraint is deliberate. The only published magnitude found for expert-imbalance cost
is a vendor blog's unverified *"recovers the 20–40% of MoE GPU cycles lost to hot
experts"* — not a primary measurement. Turning that into a coefficient would put an
unsourced number into the engine **and double-count**, because `_SPAN_BW_SHARE_EXP` is
fitted to a real rack curve that already contains that deployment's imbalance. So the cost
stays in the measured exponent, and this function names the risk, quantifies only the
balls-in-bins spread that *is* computable (a lower bound — real routers are skewed, which
is why DeepSeek ships EPLB), and cites what it cannot. Assertion **a25** pins it read-only.

## Row-by-row: the 43 changed baseline rows

Baseline rows are **replicated mode only**, so the shared-mode improvements that matter
most for the frontier entries are not represented here at all — a pre-existing limitation
of the harness worth knowing.

| model | system | rows | copies/node | tok/s/user | nodes | verdict |
|---|---|---|---|---|---|---|
| Mixtral-8x7B | Pro Max GB10 | 3 | — | 1200 → 0 | 1→0, 8→0, 64→0 | Tight Fit → **Cannot fit** |
| Mixtral-8x7B | XE9680 H200 ×8 | 6 | 11 → 8 | — | 6→8 | reserve |
| Qwen2.5-7B | Pro Max GB10 | 4 | 8 → 7 | 424–537 → 454–600 | 1→2, 8→10 | reserve + co-residency |
| Qwen2.5-7B | XE7740 RTX Pro 6000 | 6 | 12 → 10 | 2834–3470 → 3470–4908 | 1→2, 2→3 | reserve + co-residency |
| Qwen2.5-7B | XE9680 H200 ×8 | 6 | 74 → 64 | 7033–8614 → 7460–21099 | — | reserve + co-residency |
| TinyLlama-1.1B | Pro Max GB10 | 6 | 53 → 47 | 165–250 → 212–300 | 3→4 | reserve + co-residency |
| TinyLlama-1.1B | XE7740 RTX Pro 6000 | 6 | 39 → 35 | 1111–1683 → 1227–1735 | — | reserve + co-residency |
| TinyLlama-1.1B | XE9680 H200 ×8 | 6 | 468 → 416 | 2770–7460 → 7460–21099 | — | reserve + co-residency |

Two patterns account for all of it:

1. **`copies/node` falls ~10–11% everywhere** — the serving reserve, uniformly.
2. **`tok/s/user` rises at partial load** — the co-residency fix. The old model billed a
   lightly loaded system as if it were fully packed; at full packing old and new agree to
   within the reserve.

**`Mixtral-8x7B` on a GB10 at 131,072 tokens is the one verdict change**: 101.2 GB of
weights plus 17.2 GB for a single session's KV against 115.2 GB usable. It genuinely
cannot serve that context, and the tool now says so. At 512 tokens it still fits.

## Left alone, on purpose

- **The `tps_per_user` formula itself.** It treats one GPU's bandwidth as streaming the
  whole active weight set, which is physically wrong — real TP shards the read — but the
  missing TP speedup and the missing TP sync cost nearly cancel, and the result is now
  externally validated at 1.2% (a19) and across the curve (a22). Fixing one half alone
  would break a validated result. Assertions a19/a22 exist to catch exactly that.
- **Prefill / chunked prefill.** The engine is decode-only. Chunked prefill steals decode
  bandwidth and would raise $/MTok for long-context workloads. Real, unmodelled, and a
  larger change than this one.
- **Router headroom at the fleet level.** `nodes_needed = ceil(n_users / sessions_per_node)`
  assumes a perfect router filling every node to 100%. Real deployments provision slack to
  hold p99 latency. Adding it means choosing a target utilisation, which is a product
  decision rather than a physics one — it belongs in the UI as a parameter, not baked in.
