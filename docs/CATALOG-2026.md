# Model catalog refresh — 2026-11-05

## Why

The suite benchmarked 2024-era **dense** models — TinyLlama-1.1B, Llama-3.2-3B,
Qwen2.5-7B/14B/32B, Mistral-7B-v0.1, Phi-4, Mixtral-8x7B — with one modern
entry, `nvidia/Qwen3-8B-NVFP4`. Meanwhile every other project on this box
serves 2026 **MoE at FP8/FP4**.

Dense BF16 is the worst case for 128 GB at 273 GB/s. A catalog of only dense
models showcased the hardware's weakest axis while the box's own sibling
projects demonstrated its strongest.

## Added

Four cost nothing to add: sibling projects already hold the weights, and
`models/llm-models/` now carries symlinks to their snapshots.

| Catalog name | Source | On-box evidence |
|---|---|---|
| `Qwen3.6-35B-A3B` | symlink -> gb10-markets | 61 tok/s single-stream with MTP (53.8 without), 84% draft acceptance, peak 56.2 GiB at 4-way, 10.0 KiB/token KV |
| `Nemotron-3.5-Lightning-30B-A3B` | symlink -> gb10-nemotron | NVIDIA positions it *for DGX Spark*; 3.0 KiB/token KV, 1M context |
| `Laguna-S-2.1` | symlink -> gb10-laguna | 92.9 GiB measured on disk, **not** the published ~71 GB; 18 tok/s |
| `Qwen3.8-27B` | symlink -> HF cache | dense-at-FP8 control, so MoE-vs-dense is like-for-like on precision |
| `Gemma-4-26B-A4B` | downloaded, 18.8 GB | small-MoE control; community report gives 52 tok/s on GB10 |
| `Nemotron-3-Super-120B-A12B` | downloaded, 80.4 GB | NVIDIA publishes a DGX Spark deployment guide; was already projected in the TCO catalog with no way to measure it |

Every architecture figure in `MODEL_ARCH` was read from the checkpoint's own
`config.json`, not from published specs. That mattered: all four of the newest
models are hybrids whose KV lives on a minority of layers, and all four have a
`head_dim` that is **not** `hidden / q_heads` (Laguna is 3072/48 = 64 derived
vs 128 real). See `docs/KV-MODEL-CHANGE.md`.

## Retired from the benchmarkable list

| Model | Weights on disk | Why |
|---|---|---|
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 87 GB | 2023 MoE; superseded as the MoE exemplar by any A3B |
| `Qwen/Qwen2.5-32B-Instruct` | 62 GB | dense 32B BF16 is the anti-showcase; also duplicated in the HF cache |
| `mistralai/Mistral-7B-v0.1` | 28 GB | stored at 28 GB for a 7B model, i.e. FP32, and it is a *base* model, not instruct |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2.1 GB | toy; the ladder's bottom rung is Llama-3.2-3B |

Plus two unreferenced narrator-dir checkpoints: `Qwen--Qwen3.5-35B-A3B` (67 GB
BF16) and `Sehyo--Qwen3.5-35B-A3B-NVFP4` (24 GB), both superseded by the
Qwen3.6-35B-A3B-FP8 symlink.

**Removal from the list is not removal from `MODEL_CATALOG`.** `TinyLlama-1.1B`
and `Mixtral-8x7B` are named in `scripts/smoke_tco_capacity.py` and in all 90
golden-baseline rows; their catalog entries stay. `MODEL_CATALOG` is explicitly
inclusive of models too large to run here, so projection-only entries are
normal. Only `config/model-lists.json` and the UI selector changed.

## Kept deliberately

`nvidia/Qwen3-8B-NVFP4` (the only existing FP4 proof — 36.4 vs 13.1 tok/s is
the suite's cleanest result), `unsloth/Llama-3.2-3B-Instruct` (the on-device
narrator, `on_device_ai.py:25` — removing it silently disables narration),
`Phi-4`, `Qwen2.5-7B-Instruct` (the measured vLLM reference anchor) and
`Qwen2.5-14B-Instruct` (the dense ladder).

## The UI selector is now presence-filtered

`config/model-lists.json` is the single source of truth, and the selector
offers only entries that `_local_path()` actually resolves on disk, naming any
that are missing in a caption. Previously the list was hardcoded in
`streamlit_app.py`; a retired or half-downloaded entry would have made
`from_pretrained()` fall back to the HF id and pull tens of GB mid-demo.

## A safety bug found on the way in

`mem_guard.params_b()` matches lowercase substrings and falls back to 7.0B.
Four of the six additions defeated it — `Laguna-S-2.1` and
`Nemotron-3-Super-120B` matched nothing, and `Qwen3.8-27B` matched `"7b"`
inside `"27b"`. `fits_in_memory()` therefore put Laguna at 15.4 GB BF16 and
returned **fits=True** for a 92.9 GiB checkpoint, so the UI would have
attempted the load and over-committed the unified pool — the same failure that
hard-hung this box on 2026-06-30. Fixed with explicit entries ahead of the
generic suffixes; Laguna now estimates 260.3 GB and is refused.

---

# Cloud pricing refresh — 2026-11-05

## What was wrong

Every Claude price in `cloud_pricing.py` was **correct for the model it named** —
Opus 4.8 $5/$25, Sonnet 4.6 $3/$15, Haiku 4.5 $1/$5 all verified. The problem was
that the models were a generation behind, and the current generation is *cheaper*:

| Was quoted | $/1M | Current equivalent | $/1M | Delta |
|---|---|---|---|---|
| Claude Opus 4.8 | 5.00 / 25.00 | Claude Opus 5.5 | **4.00 / 20.00** | 20% cheaper |
| Claude Sonnet 4.6 | 3.00 / 15.00 | Claude Sonnet 5.5 | **2.00 / 10.00** | 33% cheaper |
| Claude Haiku 4.5 | 1.00 / 5.00 | Claude Haiku 5.5 | **0.10 / 0.50** | **10x cheaper** |
| — | — | Claude Fable 5.1 | 10.00 / 50.00 | no row existed |

So the table overstated hosted-API cost — which flattered on-prem by accident.
That is the same failure mode as an unlabelled projection: a number that leans
one way without saying so.

## The stance, stated

The comparison is now **deliberately conservative toward on-prem, and says so on
screen** (`STANCE_LABEL`, rendered as a warning banner above the comparison).
Four levers, all of them real published pricing structure:

1. **Long-context tiers applied.** Claude Haiku 5.5 is 5x above a 100K prompt;
   Gemini 3.1 Pro is ~2x above 200K. The TCO tab **defaults to 131,072 context**
   and goes to 524,288, so the higher tier is the normal case here, not an edge
   case. The previous module ignored tiers entirely and therefore understated
   cloud cost at every default setting — this lever makes it *more accurate* and
   higher at the same time.
2. **Thinking tokens billed as output.** Adaptive thinking is on by default on
   every current model, so a reasoning workload is billed for more output than it
   returns. `THINKING_OUTPUT_MULTIPLIER = 1.5` is the **only estimated input in
   the module** and is labelled as an assumption, not a measurement; set it to
   1.0 for a thinking-free workload.
3. **Frontier-class price point.** `CLOUD_ANCHORS` moved from
   `point = Gemini 3.5 Flash` / `high = Claude Sonnet 4.6` (Sonnet is mid-tier,
   mislabelled "frontier") to `point = Claude Opus 5.5` / `high = Claude Fable 5.1`,
   with `low = Claude Haiku 5.5` keeping the honest floor visible.
4. **Cache reads and the 50% Batch discount are NOT applied.** Both are real and
   both would lower cloud cost. Omitting them is conservative and is recorded so
   the omission is visible rather than accidental.

Effect at the default 131,072 ctx / 16,384 out: the quoted price point goes from
**$17.00** to **$41.33 per 1M output tokens**, with a **$103.33** ceiling — every
figure a published list price.

## What was NOT done, deliberately

**The stale Haiku 4.5 rate was not kept just because it was higher.** At its real
>100K tier Haiku 5.5 costs $5.17/1M out versus the stale Haiku 4.5's $10.33 — so
correcting it *lowered* the floor. Using a wrong number because it favours the
conclusion is fabrication, and it is the first thing a customer's procurement
team would catch. The stance comes from pricing structure, never from stale data.

**Gemini rows were not changed.** Third-party trackers broadly agree with the
rates already in the file (3.1 Pro $2/$12, 3.5 Flash $1.50/$9), but Google's own
pricing page could not be reached on 2026-11-05, so every Gemini row now carries
`verified: False` and the UI says the Gemini rates are unverified. One live risk
is recorded: Gemini 3.8 Flash is reported to be on an introductory rate ending
**2027-01-01**, which would silently invalidate the comparison on New Year's Day.

## Also verified, not changed

The box's own Claude spending is correctly priced. `second_opinion.py:23` carries
`PRICE = {"input": 5.00, "output": 25.00}` for `claude-opus-5`, which matches the
first-party rate; the four logged second opinions ($0.065-$0.09) are accurate.
Moving that path to `claude-opus-5-5` would cut it ~20% and require the constant
to become 4.00/20.00.

---

# C0 downloads complete — 2026-11-05 12:03

All four exact manifest matches, `rc=0`, 1 h 36 m total:

| Repo | On disk | Expected |
|---|---|---|
| `nvidia/Gemma-4-26B-A4B-NVFP4` | 18.8 GB | 18.8 GB |
| `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` | 24.1 GB | 24.1 GB |
| `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` | 75.1 GB | 75.1 GB |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 80.4 GB | 80.4 GB |

## Nemotron-3-Super is 8 attention layers of 88

Its config carries no `layer_types`, but it does carry
`hybrid_override_pattern` — the NemotronH convention where `*` is attention,
`M` is mamba and `E` is MLP/MoE:

```
MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME
```

**8 attention, 40 mamba, 40 MoE.** Billing all 88 layers would have over-stated
its KV by **11x**. At fp8 it is **4.0 KiB/token** — a 120B model with less KV per
token than a 3B dense one, because 80 of its 88 layers hold no growing cache.

With that row added, **no LLM catalog entry is on the legacy KV fallback.**

## Measured checkpoint size now beats the flat estimate

A flat bytes-per-param figure is wrong for mixed-precision NVFP4 releases, which
commonly quantize only the expert projections and leave the rest BF16:

| Model | flat 0.5 B/param | measured on disk | ratio |
|---|---|---|---|
| Laguna-S-2.1 | 59.0 GB | **92.9 GiB** | 1.6x |
| Nemotron-3-Super-120B | 60.0 GB | **80.4 GB** | 1.3x |

`MODEL_CATALOG` entries now take an optional `disk_gb` per precision, and
`_weights_and_raw()` prefers it. Under-stating weights on a 121 GB unified pool
is the dangerous direction — and the corrected figures match observed reality:

- **Laguna**: 108.1 GB weights + 6.4 GB KV at 256K = **114.5 GB against 100 GB
  usable → will not fit.** That matches the live box, where Laguna needs
  `--gpu-memory-utilization 0.92` and currently fails to start.
- **Nemotron-3-Super**: 87.1 + 1.1 = **88.1 GB → fits, but tightly**, matching
  the "monitor memory closely" caution on a 120B NVFP4 on this hardware.

## model_info.py gap closed

The C2 commit added six models to the catalog without rows in
`model_info.py`'s `MODEL_INFO` / `DESIGNED_PRECISION`, so the UI's detail panel
and `precision_compatible()` had nothing for them. All six now have entries
carrying the measured footprint, the real attention-layer split, and the
per-model footguns (triton MoE backend for Qwen3.6; Bearer-only auth and
explicit `enable_thinking` for Laguna). **0 catalog entries missing either
table.**

Final state: LLM 11/11 on disk and offered, VLM 6/6 on disk, harness 12/12,
app serves HTTP 200.

---

# TCO tab restricted to benchmarkable models — 2026-11-05

The TCO "Target Model" selector offered all 38 `MODEL_CATALOG` entries, most of
which this box cannot run. It now defaults to the **11 models present on disk**,
so every TCO figure can be grounded in a measurement taken here rather than a
projection. The hidden 27 entries (405B, 671B, rack-scale) stay available behind
an explicit checkbox labelled as projection-only — deleting them outright would
remove the "what hardware would I need for DeepSeek 671B" answer and leave the
rack systems in `DELL_SYSTEMS` with nothing to size.

`BENCHMARKABLE` in `tco_engine.py` maps HF repo id -> catalog key **explicitly**.
Substring matching is what made `mem_guard` report a 118B checkpoint as 7B, and
a wrong match here would silently corrupt every TCO figure for that model.

The list is computed from `config/model-lists.json` filtered by on-disk presence,
via a new module-level `_llm_catalog_ids()` helper — **not** from the sidebar's
`llm_models`, which only exists inside the LLM-scenario branch and would have
raised `NameError` in the TCO tab under a vision or HPC scenario.

## What this unlocked, visible immediately

All 11 at 8 users / 131K context on one GB10, shared-weights capacity:

| Model | nodes | sessions/node | recommendation |
|---|---|---|---|
| Nemotron-3.5-Lightning-30B-A3B | 1 | **138** | Single-GPU |
| Nemotron-3-Super-120B | 1 | **37** | Single-GPU |
| Qwen3.6-35B-A3B | 1 | 33 | Single-GPU |
| Gemma-4-26B-A4B | 1 | 21 | Single-GPU |
| Qwen2.5-7B | 1 | 14 | Single-GPU |
| Qwen3.8-27B | 1 | 11 | Single-GPU |
| Llama-3.2-3B | 1 | 8 | Single-GPU |
| nvidia/Qwen3-8B-NVFP4 | 2 | 6 | Single-GPU |
| Phi-4 (14B) | 3 | 3 | Single-GPU |
| Qwen2.5-14B | 3 | 3 | Single-GPU |
| Laguna-S-2.1 | 3 | 3 | Tight Fit |

A **120B** MoE serves 37 concurrent sessions on one box while a **14B** dense
model needs three boxes to serve 3 each. Before the KV fix and the catalog
refresh this ordering was impossible to see, because capacity tracked parameter
count and nothing else.

## Known limitation until the vLLM re-capture (C5)

The `$/MTok` column reads **$0.250 for seven of the eleven models**. That is not
a coincidence and not a bug in the capacity model — every row was computed with
the same `gb10_tps` input (61.0, the figure measured for Qwen3.6-35B-A3B), because
the reference capture holds only two models, both <=8B dense, from 2026-07-21.

So right now: **capacity per node is per-model and correct** (it comes from
attention geometry and measured checkpoint size), while **throughput is not
differentiated** — it is one anchor applied to every model. C5 adds the MoE
anchors and the MTP on/off pair; the deferred `active_b` decode term then makes
the transfer between models honest. Until both land, read sessions/node and
nodes as real and treat `$/MTok` as provisional.

---

# Vera Rubin NVL72 added as XE9812 — 2026-11-05

Priced at the supplied **$8,724,954 per rack** (GPU-inclusive, like the other
rack-scale rows; the $121,180/GPU figure is a modelling reference and is never
added).

## The GPU spec needed no change

`GPU_SPECS["Vera Rubin / VR200"]` already held per-GPU figures that reproduce
every rack number in the supplied stats when multiplied by 72:

| Rack figure | Supplied | 72 x per-GPU spec | |
|---|---|---|---|
| FP4 inference | 3.6 EF | 72 x 50 PFLOPS = 3.6 EF | match |
| NVFP4 training | 2.5 EF | 72 x 35 PFLOPS = 2.52 EF | match |
| FP8 training | ~1.2 EF | 72 x 17.5 PFLOPS = 1.26 EF | match |
| HBM capacity | 20.7 TB | 72 x 288 GB = 20.736 TB | match |
| HBM bandwidth | 1.6 PB/s | 72 x 22 TB/s = 1.584 PB/s | match |
| NVLink scale-up | 260 TB/s | 72 x 3.6 TB/s = 259.2 TB/s | match |

The dense/sparse convention already in the file (`FP4: _tf(35_000, 50_000)` —
dense = training peak, sparse = inference peak) is exactly the 10-vs-20 /
35-vs-50 split in the stats, and matches how GB300 is entered.

## One figure confirmed, two still inferred

Called out in the code so they can never be mistaken for published data:

1. ~~Rack power~~ — **CONFIRMED at 230 kW on 2026-11-05.** This is the figure,
   not an estimate, and it should not be "corrected" against a blog. For context
   on why that matters: NVIDIA does not publish a rack power number publicly and
   third-party sources disagree badly — supply-chain analysis quotes 190 kW
   Max-Q / 230 kW Max-P, GIGABYTE's own NVL72 build lists a 240 kW maximum, some
   articles still repeat Blackwell-era 120-130 kW, and the "600 kW" figure
   belongs to Rubin Ultra / Kyber rather than VR200. GB300 NVL72 is 136 kW, so
   230 kW is ~1.7x the power for ~2.5x the FP4 inference throughput.
2. **Scale-out networking 28,800 Gb/s**, carried over from XE9712 unchanged.
   Vera Rubin is expected to pair with ConnectX-9, but no scale-out figure was
   supplied. The 260 TB/s NVLink number is scale-**up** (inside the rack) and is
   modelled by `nvlink_gbs`, not by `nw`.
3. **Generation "18G"**, inferred from the platform naming pending Dell's own
   matrix.

`nvlink-switch` is reused as the link class rather than inventing a better
`nvlink6` penalty. NVLink 6 does double rack scale-up, but we have no measured
parallel-efficiency figure for Rubin, and crediting the newest part with an
unmeasured improvement would flatter it. **This under-states Rubin on purpose.**

## What the model produces

Qwen3-235B-A22B, 512 users, 131K context, shared-weights capacity:

| | GB300 NVL72 | Vera Rubin NVL72 | ratio |
|---|---|---|---|
| rack bandwidth | 576,000 GB/s | 1,584,000 GB/s | 2.75x |
| rack power | 136 kW | 230 kW | 1.69x |
| rack price | $6,200,000 | $8,724,954 | 1.41x |
| aggregate tok/s | 853,934 | 2,348,318 | 2.75x |
| **$/MTok** | **$0.3012** | **$0.1557** | **1.93x better** |
| tok/s per MW | 6.28M | 10.21M | 1.63x |

**Sessions per node is identical (180) and that is correct, not a bug** — HBM
capacity is essentially unchanged between the two racks (20.7 TB both), exactly
as the supplied stats state. Rubin's entire advantage here is bandwidth, which
is what decode is bound by, so it shows up in throughput and $/MTok rather than
in concurrency.

**The vendor "up to 30x throughput per megawatt" claim is not reproduced.** The
model gives **1.63x**, from 2.75x bandwidth against 1.69x power. The 30x and the
"35x lower token cost" figures are workload-specific vendor estimates and are
deliberately not modelled; the GPU spec notes say so. The supplied stats also
warn that the "8 exaFLOPS / 7.5x GB300" number refers to the **NVL144 CPX**
variant, which is a different system and is not in this catalog.

## Verified against the supplied GB300 / VR200 comparison table (2026-11-05)

**10 of 13 rows match.** Three do not, listed by whether they affect a TCO number.

| Row | GB300 want | GB300 have | VR200 want | VR200 have | |
|---|---|---|---|---|---|
| FP4 inference | 1.44 EF | 1.44 EF | 3.6 EF | 3.6 EF | ok |
| FP4 inference / GPU | 20 PF | 20 PF | 50 PF | 50 PF | ok |
| FP4 training | 0.72 EF | — | 2.5 EF | 2.52 EF | see below |
| FP8 training | 0.36 EF | 0.36 EF | 1.2 EF | 1.26 EF | 5% |
| GPU memory | 20 TB | **20.736 TB** | 20.7 TB | 20.736 TB | conflict |
| HBM bandwidth | 576 TB/s | 576 TB/s | 1.6 PB/s | 1.584 PB/s | ok |
| Bandwidth / GPU | 8 TB/s | 8 TB/s | 22 TB/s | 22 TB/s | ok |
| NVLink | 130 TB/s | 129.6 TB/s | 260 TB/s | 259.2 TB/s | ok |
| Rack power | ~140 kW | **136 kW** | 230 kW | 230 kW | conflict |
| GPUs / CPUs, CPU cores, process node | — | now in notes | — | now in notes | added |

### Affects TCO — needs a decision

1. **GB300 rack HBM: 20 TB vs 20.736 TB.** The catalog holds 288 GB/GPU x 72 =
   20.736 TB. `vram_gb` feeds capacity -> sessions/node -> nodes -> CapEx ->
   $/MTok, so this one is load-bearing. **Kept at 20.736 TB**, because 288 GB/GPU
   is the published per-GPU figure for Blackwell Ultra and 20 TB looks like a
   rounded rack headline. Note both racks are 288 GB/GPU parts, which is exactly
   why the supplied notes say capacity "barely changes" — at 288 GB/GPU it does
   not change at all. If 20 TB is authoritative, per-GPU would have to be
   277.8 GB, contradicting the 288 GB spec.
2. **GB300 rack power: ~140 kW vs 136 kW.** Feeds `annual_power_usd` (~3% of
   rack opex). **Kept at 136 kW** — the specific published figure, and the
   supplied value carries a "~". Changing it makes GB300 look ~3% worse on opex,
   i.e. flatters Rubin, so leaving it is the conservative choice.

### Display only — no TCO impact

3. **VR200 FP8 training: 1.2 EF vs 1.26 EF.** 17.5 PFLOPS/GPU x 72 = 1.26 EF;
   1.2 EF implies 16.67 PFLOPS/GPU. Within 5%, so one side is rounded. Left as is.
4. **The FP4 dense slot does not mean the same thing on the two racks.**
   `_tf(dense, sparse)` normally means dense vs *structured sparsity*:
   - GB300 holds true dense/sparse (15 / 20 PFLOPS per GPU)
   - VR200 holds training/inference peaks (35 / 50 PFLOPS per GPU)

   The supplied table makes this visible: GB300 FP4 **training** is 10 PFLOPS/GPU
   (0.72 EF rack), which is neither its dense 15 nor its sparse 20 — so training
   is a **third axis**, and VR200's dense slot is carrying a training number.
   This predates the Rubin row (the original comment in the file already said
   "FP4 dense = NVFP4 training peak"). It is harmless for costing because
   **`calculate_tco()` never reads `tflops`** — `aggregate_tflops()` only feeds
   the spec panels — but the dense slots must not be compared across parts. Both
   entries now say so, and GB300's training peaks (FP4 10 PF/GPU, FP8 5 PF/GPU)
   are recorded in its notes.

### Added from the table

GPUs/CPUs (72 Blackwell Ultra / 36 Grace vs 72 Rubin / 36 Vera), CPU cores
(72 vs 88 per CPU), and process node (TSMC N4 vs 3nm) were absent from the
catalog and are now in both `notes` strings.
