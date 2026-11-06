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

---

# "Memory is not free" on the Benchmark tab — 2026-11-05

## Cause: my own downloads, not a real shortage

`llm_inference.py:134` guards every load against
`torch.cuda.mem_get_info()[0]`, which on this unified-memory box reports
**MemFree, not MemAvailable**. After the 198 GB of C0 downloads, ~98 GiB sat in
**reclaimable page cache**, so the guard saw 12.8 GiB free on a box with
113.6 GiB available and refused anything above ~11 GB:

```
MemTotal     121.6 GiB      mem_get_info free = 12.8 GiB
MemFree       13.3 GiB      Qwen2.5-7B  FP16 needs 17.0 GB -> BLOCKED
MemAvailable 113.6 GiB      Qwen3.6-35B FP8  needs 42.5 GB -> BLOCKED
Cached        98.2 GiB
```

Order confirmed: the guard (line 134) runs **before** `_warm_file_cache`
(line 161), so per-model cache warming is not self-blocking. The cache was
purely download residue.

## Fix: `scripts/drop_model_page_cache.py`

`posix_fadvise(POSIX_FADV_DONTNEED)` on every weight file under the model roots.
It evicts only **clean** pages (read-only weights), writes nothing, loses
nothing, and needs no root. The cache repopulates on the next read, which
`_warm_file_cache()` does deliberately before each benchmark anyway.

```
advised 699 weight files, 715.3 GB of file data
Cached    98.2 ->   3.4 GiB   (-94.8)
MemFree   13.3 -> 110.6 GiB   (+97.3)
```

All ten benchmarkable precisions pass afterwards, from Llama-3.2-3B at FP16
(7.3 GB) to Nemotron-3-Super at FP4 (72.8 GB). **Run this after any large
model download.**

## A safety gap this exposed, now fixed

`mem_guard.params_b` x bytes-per-param under-stated the two mixed-precision
NVFP4 checkpoints, because those releases quantize only the expert projections:

| Model | flat estimate | measured on disk |
|---|---|---|
| Laguna-S-2.1 | 59.0 GB | **92.9 GiB** |
| Nemotron-3-Super-120B | 60.0 GB | **80.4 GB** |

`tco_engine` already preferred the measurement (via `disk_gb`); `mem_guard` did
not, so the Benchmark tab would have let Laguna attempt a load believing it
needed 65 GB. `_MEASURED_GB` now mirrors `disk_gb` and applies at FP4/NVFP4
only — asking for BF16 on a pre-quantized checkpoint is a precision-compat
error handled elsewhere and must not inherit the FP4 file size.

Effect: **Laguna now estimates 110.1 GB and is correctly REFUSED** against the
100 GB usable window — which matches the live box, where it needs
`--gpu-memory-utilization 0.92` and currently fails to start. Nemotron-3-Super
estimates 88.7 GB and passes, tightly. No change to any other model
(Qwen2.5-7B 15.4 GB, Qwen3-8B-NVFP4 4.4 GB, Phi-4 30.9 GB all unchanged).

## What was deliberately NOT changed

The guard still reads `MemFree` rather than `MemAvailable`. Switching it would
have made this symptom disappear, but it would also make the guard trust
reclaimable memory that another allocation could claim first — and this guard
exists because an over-commit on this box **deadlocks the kernel rather than
raising a clean OOM** (2026-06-30, hard power reset). That is a risk decision
with a real failure mode behind it, not a bug fix to slip in while debugging a
benchmark, so it stays as it is pending an explicit call.

---

# FP4 path fixed + narration reuses the benchmarked model — 2026-11-05

## Why FP4 results were labelled "BF16 compute" (correct, and still are)

Two FP4 paths exist and **neither does real FP4 math**:

* **Pre-quantized NVFP4** checkpoints — compressed-tensors decompresses to BF16
  for the matmul.
* **bitsandbytes FP4** — `bnb_4bit_compute_dtype=torch.bfloat16`: 4-bit storage,
  BF16 compute.

The `"BF16 compute"` label is accurate and the UI's red *"FP4: BF16 COMPUTE"*
badge is right. **Real NVFP4 tensor-core math on this box exists only through
vLLM**: 36.4 tok/s against 13.1 for a BF16 7B, captured in
`webui/data/vllm_reference.json` with `fp4_supported: true`.

## But the FP4 path was misconfigured — 2.7x too much memory

`llm_inference.py` carried a comment claiming `run_compressed=True` needed
python3-dev headers "absent on this box". **python3.12-dev is installed**
(3.12.3-1ubuntu0.17), so the claim was stale and the packed path loads fine.

Measured on `nvidia/Qwen3-8B-NVFP4`, 3 x 128 tokens per arm:

| path | resident | tok/s (median of 3) |
|---|---|---|
| `run_compressed=False` (was in use) | **16.07 GiB** | 6.03 |
| `run_compressed=True` (now in use) | **5.96 GiB** | 6.01 |

**Identical throughput, 2.7x less memory.** The old path expanded a 6 GB
checkpoint into 16 GB of BF16 and bought nothing — so the FP4 row was claiming
a quantization story while delivering neither the speed nor the memory win.
Weights now stay packed (`uint8` + `float8_e4m3fn` scales), verified through the
real `LLMInference` path at 5.96 GiB.

A correction worth recording: a first pass measured 6.1 vs 7.3 tok/s from single
32-token runs and read as "also 20% faster". The 3 x 128 A/B shows that was
noise. Same speed, less memory, nothing more.

The load falls back to the BF16-decompress path (with a printed warning) if the
packed load raises, so a checkpoint that cannot do it still benchmarks.

## Talking points now reuse the last benchmarked model

Previously every LLM run ended in `inf.unload()`, and clicking Deep dive then
loaded a separate Llama-3.2-3B narrator — a second model's memory plus an
unload/load cycle, for prose the resident model can write just as well.

* `OnDeviceNarrator(inf=...)` adopts an already-loaded `LLMInference`, tracking
  `owns_inf` so `unload()` never frees a model the Benchmark tab still holds.
  `model_label` reports which model is writing, e.g.
  *"Qwen2.5-7B-Instruct @ FP8 (reused from benchmark)"*.
* `retain_for_narration(inf)` replaces the per-run unload in the LLM paths. Only
  **one** model is ever retained — a previous one is unloaded first — and the
  next run calls `unload_narrator_for_benchmark()`, which now releases the
  retained model too, so benchmarks still start from a clean GPU.
* The **Vision** path still unloads: `VisionModelInference` has no
  `generate_stream()`.
* Fallback is unchanged — with nothing resident, the dedicated narrator loads as
  before.

Verified: adopting a resident Qwen2.5-7B @ FP8 produced talking points with a
memory delta of **+0.00 GiB**, and `narrator.unload()` left the benchmark model
loaded.

### A latent over-claiming path closed on the way

`_fmt_metrics()` reads `BenchmarkMetrics` **objects** via `getattr`. Handed
anything else it degrades to `precision=? · bottleneck=?`, and a model told to
"use ONLY the numbers provided" then **invented a throughput figure** (a test
with dict-shaped metrics produced "approximately 260 queries per second" from no
data at all). `results_summary()` now refuses to call the model when there are no
usable numbers and returns a notice instead. An empty panel is recoverable; a
fabricated benchmark number in front of a customer is not.

### Tradeoff to know

Narration quality now varies with whatever was last benchmarked. A 3B at FP16
writes differently from a 35B MoE, and a weaker model may hedge more. The
numbers it is given are always the measured ones, and the guard above stops it
narrating with none.

---

# Cloud panel: hosted-options table removed, on-prem comparison added — 2026-11-05

**Removed** the "All hosted options for reference" caption and its 12-row
dataframe. The low/price-point/high band above it already carries the range, and
the full table invited reading a cheap Gemini tier as the comparison basis.
`_api_costs` is still computed — the narrator's `cloud_comparison()` uses it.

**Fixed a stale caption** found while editing: the footer still read
*"Low = Gemini 2.5 Flash · price point = Gemini 3.5 Flash · high = Claude Sonnet
4.6"* — the anchors before the pricing refresh moved them to Haiku 5.5 / Opus 5.5
/ Fable 5.1. It directly contradicted the table above it. It is now **generated
from `CLOUD_ANCHORS`**, so it cannot drift again, and it names which providers
are unverified.

**Added the comparison that was missing.** The tab computed a cheapest on-prem
`$/MTok` and a cloud `$/MTok` and never put them next to each other — the only
link was a passing mention in a caption. Now: three metrics (cheapest viable
on-prem platform, cloud price point, and the ratio) plus a verdict line.

## The ratio needed qualifying, not just displaying

First run produced **1,744x** (Qwen3.6-35B-A3B, 64 users, 131K ctx: $0.0237 vs
$41.33 per 1M output tokens). That is an artefact, not a result, and quoting it
unqualified is how a TCO deck loses the room. Two conditions produce it, so both
are now stated **with** the number rather than in a footnote:

1. **Utilisation.** The on-prem figure amortises the entire platform over the
   full term *while serving the configured concurrency continuously*. Idle
   hardware costs the same and produces no tokens, so the honest claim is
   "N x cheaper **at full utilisation**, scaled by your actual duty cycle".
2. **Capability.** The price point is a frontier model, very likely more capable
   than the local model being costed. The panel now also shows the **low anchor
   as the like-for-like floor** — Haiku 5.5 at $5.17/1M out, still 218x the
   on-prem cost. Stated plainly: this compares cost per token, not capability
   per token.

For reference at that workload: low 218x · price point 1,744x · high 4,360x.
The direction is not in doubt; the multiple depends entirely on duty cycle and
on which cloud model is a fair substitute.

Omissions are listed on both sides so neither looks accidentally clean —
on-prem: staff, floor space, cooling beyond rack power, CapEx opportunity cost,
and any duty cycle below 100%; cloud: cache reads and the 50% batch discount.

---

# NVFP4 "failed to load" — ModelOpt checkpoints are not HF-loadable — 2026-11-05

## What was actually wrong, and it was mine

Three of the six models I added this morning **cannot be benchmarked through the
HF path at all**, and I added them without verifying a load. Worse than failing,
one of them *succeeded* and produced nonsense.

`transformers 5.5.4` implements `compressed-tensors`, `fp8`, `torchao`, `mxfp4`
and ~18 others — but **not NVIDIA ModelOpt**. Given a `modelopt` checkpoint it
prints *"Unknown quantization type, got modelopt … Hence, we will skip the
quantization"*, reinterprets the packed uint8 weights as BF16, and carries on.

Measured on `nvidia/Gemma-4-26B-A4B-NVFP4`:

```
load_ok   = True
resident  = 48.07 GiB          (a 18.8 GB checkpoint)
dtypes    = {bfloat16: 426}    (packed weights reinterpreted)
generated = '0-1 way/radius housing t[i deる'
```

**A successful-looking load, 48 GB of memory, and garbage output.** A benchmark
would have recorded a throughput figure from a broken model — the one result
this suite must never produce.

`Nemotron-3.5-Lightning` and `Nemotron-3-Super-120B` fail more honestly: the
shapes don't match and the load raises `ignore_mismatched_sizes`, which is the
"NVFP4 failed to load" that was reported.

## Audit of every quantized checkpoint

| Checkpoint | quant_method | HF |
|---|---|---|
| Qwen3.6-35B-A3B-FP8 | `fp8` | ok |
| Qwen3.8-27B-FP8 | `fp8` | ok |
| Qwen3-8B-NVFP4 | `compressed-tensors` | ok |
| Laguna-S-2.1-NVFP4 | `compressed-tensors` | ok (refused on memory: 110 GB) |
| Gemma-4-26B-A4B-NVFP4 | **`modelopt`** | **silently wrong** |
| Nemotron-3.5-Lightning-30B-A3B | **`modelopt`** | **raises** |
| Nemotron-3-Super-120B-A12B | **`modelopt`** | **raises** |
| Nemotron-3.5-Bolt-74B-A9B | **`modelopt`** | **unsupported** |
| Nemotron-3.5-Super-VL-120B-A12B | **`modelopt_mixed`** | **unsupported** |

## Fix: refuse before loading, and say why

`precision_compatible()` now reads the checkpoint's declared `quant_method` from
`config.json` (cheaply, without importing remote code) and refuses anything the
installed transformers cannot honour, reusing the existing **Not Compatible**
treatment rather than inventing a new failure mode:

> ⛔ checkpoint is quantized with 'modelopt', which transformers 5.5.4 does not
> implement — it would silently skip quantization, load as BF16 and generate
> garbage. This model runs correctly under vLLM (which is how the sibling
> projects serve it), not through the HF benchmark path.

Verified: Gemma-4 is now refused **before** any load; `Qwen3-8B-NVFP4` still
loads packed at 5.96 GiB and generates coherently.

## Correcting an earlier claim

I previously reported that "all ten benchmarkable precisions pass". That was
wrong. It tested only the **memory guard**, not a load. Five of eleven LLM
entries do not run through HF: three ModelOpt refusals, plus Laguna refused on
memory. The honest benchmarkable set is **Llama-3.2-3B, Qwen2.5-7B,
Qwen3-8B-NVFP4, Phi-4, Qwen2.5-14B, Qwen3.8-27B-FP8 and Qwen3.6-35B-A3B-FP8**.

The ModelOpt models stay in the catalog deliberately: they are real, on disk,
correct in the TCO model, and they run under vLLM — which turns a dead end into
the point. "Our benchmark harness can't run this; the serving stack we actually
deploy can" is the vLLM-vs-HF story the honesty rework already tells about FP4.

---

# C5 — vLLM reference re-capture — 2026-11-05

## What was wrong with the old reference

`webui/data/vllm_reference.json` held two captures, both <=8B **dense**, from
2026-07-21, and the NVFP4 one was **non-monotonic**: n=8 reported 24.3 tok/s per
user against n=16's 37.1, taken at **98% KV-cache usage**. Past ~85% the pool is
full, vLLM preempts and queues, and the sweep stops measuring throughput scaling
and starts measuring cache pressure. The TCO engine calibrates
`_BATCH_BW_SHARE_EXP` against that file.

## Harness fixes this required

Three defects surfaced on the first attempt, which failed all four arms in 2.5
minutes:

1. **HF-cache snapshots are directories of symlinks.** The symlinked models point
   at `.../snapshots/<hash>/`, whose entries are 56-69 symlinks into
   `../../blobs/`. Mounting the snapshot alone left every weight dangling inside
   the container. Those arms now mount the hub ROOT with `HF_HOME` and address
   the model by repo id, exactly as `markets-vllm.service` does.
2. **`docker run --rm` destroyed the evidence.** An exited `--rm` container is
   removed before `docker logs` can run, so the failure reported "No such
   container" instead of the actual error. `--rm` dropped; `cleanup()` removes it
   explicitly.
3. **`nvidia/Qwen3-8B-NVFP4` ships no chat template**, so
   `/v1/chat/completions` returns 400. That arm uses `--mode completions`.

And one data-loss bug worth recording: the template failure was recorded as
**`fp4_supported=false`, overwriting a good capture** — which would have made the
demo claim real FP4 does not work, when it demonstrably does. Restored from git.
`--fp4` now only attributes a failure to FP4 when the error could plausibly BE an
FP4 problem; chat-template, transport and 404 errors leave the flag unset.

## New: a KV ceiling on the sweep

`capture_vllm_reference.py --kv-max` (default 0.85) stops escalating concurrency
once the KV pool is near full. Also fixed the print, which showed
`vllm:kv_cache_usage_perc` — a 0..1 fraction — with a `%` suffix, so 98% read as
"0.98%".

## Qwen3-8B-NVFP4, re-captured

| n | per-user | aggregate | KV |
|---|---|---|---|
| 1 | 39 tok/s | 39 | 12% |
| 2 | 41 | 81 | 23% |
| 4 | 40 | 161 | 45% |
| 8 | 40 | **321** | 89% -> stopped |

Monotonic, per-user flat at ~40 tok/s while aggregate scales near-linearly —
real shared-weights behaviour, and the n=8 anomaly is gone.

## Capture config is deliberately NOT the production config

`markets-vllm` serves the 35B at `--max-model-len 262144 --max-num-seqs 4`, tuned
for long-context single-user work. A concurrency sweep needs seats, so the
capture arms use a short context and 32 seats. That measures how aggregate
throughput **scales with concurrency** — the quantity `_BATCH_BW_SHARE_EXP`
models — rather than as-deployed long-context behaviour. Both are real; they
answer different questions.

## Recalibration

`scripts/recalibrate_batch_exp.py` fits the exponent by least squares on
log(aggregate) vs log(n), anchored at n=1:

| capture | levels | fitted exp |
|---|---|---|
| qwen2.5-7b (BF16) | 6 | 0.921 |
| qwen3-8b-nvfp4 (new) | 4 | **0.989** |
| pooled | | **0.955** |

Pinned value: **0.9**. The measured 0.955 is +0.055 above it, comfortably inside
assertion 5's `[0.5, 2.0]x` band. **Left at 0.9 on purpose**: a lower exponent
under-states aggregate throughput, and under-claiming capacity is the safe
direction for a number a customer may provision against. The measurement is
recorded here so raising it later is a decision with evidence rather than a tweak.

## C5 outcome — 2 of 4 arms, and a guard flaw found

| capture | precision | levels | n=1 | peak aggregate | KV |
|---|---|---|---|---|---|
| qwen2.5-7b | BF16 | 6 | 13.1 | 399.5 | 62% |
| qwen3-8b-nvfp4 | NVFP4 | **4** | 38.8 | **320.7** | 89% |
| qwen3.6-35b-a3b-fp8 | FP8 | **1** | **50.8** | 50.8 | 100% |

**The 8B re-capture is the clean win**: monotonic, 4 levels, stopped at the
ceiling instead of running into it.

**The 35B produced only a single point, and that exposed a flaw in the ceiling I
had just added.** `vllm:kv_cache_usage_perc` reported **100% at n=1**. On a
hybrid linear/full-attention model vLLM **pre-allocates** per-seat state, so the
metric reflects pool *allocated*, not pool *under pressure* — and the guard
stopped the sweep after one level, producing no scaling curve. A ceiling that can
end a sweep before it begins is worse than no ceiling. Fixed: the ceiling is only
honoured after at least 3 levels.

The single point is still worth keeping: **50.8 tok/s single-stream** against the
**53.8 tok/s** gb10-markets recorded for the same model without MTP — an
independent cross-check within 6%.

**Not captured:** the MTP arm and Nemotron-3.5-Lightning. Both need a re-run with
the fixed guard, and the 35B arms additionally need a larger `--kv-cache-memory`
or fewer seats so the pool is not saturated by pre-allocation. `$/MTok` therefore
remains single-anchored, and the deferred `active_b` decode term still has no MoE
scaling curve to calibrate against — that is the one plan item C5 has not
delivered.

---

# Pre-quantized checkpoints: only compressed-tensors works here — 2026-11-05

Testing C6 exposed that the ModelOpt refusal was **too narrow**. I had allowed
`fp8` because transformers registers an fp8 quantizer. It registers it; it does
not make it work on this stack. Every pre-quantized format was then loaded AND
generated from:

| quant_method | model | generated | |
|---|---|---|---|
| compressed-tensors | Qwen3-8B-NVFP4 | `How about: "Hello, how are you?"` | **OK** |
| fp8 (block) | Qwen3.6-35B-A3B | `upift如无一言stluanncno源halt` | garbage |
| fp8 (block) | Qwen3.8-27B | `守则ächendblلابymmarius踵alaus` | garbage |
| modelopt | Gemma-4-26B-A4B | `0-1 way/radius housing t[i de` | garbage |
| modelopt | Nemotron-3-Super / Lightning | — | raises |

Every garbage case reports a **successful load**. A benchmark would have recorded
throughput for a broken model.

One wrong theory discarded on the way: I first assumed the fp8 garbage came from
the demo stacking its own torchao FP8 pass on already-FP8 weights, and added a
pass-through so pre-quantized checkpoints load as-is. **It made no difference** —
the output was still garbage — so the cause is the stack's fp8 path itself, not
double-quantization. The pass-through is kept because re-quantizing a quantized
checkpoint is wrong regardless, but it was not the fix.

`_HF_SUPPORTED_QUANT` is now the **empirical** set — `{compressed-tensors}` —
with the measurements in the comment, so the list cannot drift back to "what
transformers advertises".

## Honest runnable set

**LLM (5):** Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B-NVFP4
(compressed-tensors), Phi-4, Qwen2.5-14B-Instruct.
**Vision encoders (4):** CLIP B/32, CLIP L/14, ViT-base, DINO-ViT-S/16.
**Refused:** 5 LLM entries on quantization, Laguna on memory, all 3 VLM entries
on quantization.

## C6 status: implemented, not exercisable

The vision track now has a real image+prompt->text path, VLM-vs-encoder
detection, the three memory guards it previously lacked, a fixture with known
values, and UI wiring that checks the answer against the truth file. `bin/check_vlm.py`
passes its 4 static assertions.

**But no VLM on this box can be loaded through HF**: the two Nemotron VL
checkpoints are ModelOpt, and the one HF-loadable candidate (Qwen3.6-35B, natively
a VLM) is block-fp8 — all three now correctly refused. So checks 5-7 report SKIP
with that reason rather than passing or failing. The capability is built and the
prerequisite is unmet; closing it needs either a compressed-tensors VLM
checkpoint or a vLLM-backed vision path, which is the same conclusion C5 reached
about ModelOpt models generally.

---

# The TCO demo does not need the HF path at all — 2026-11-05

## The realisation

`calculate_tco()` contains **zero** model-loading calls. The TCO tab needs three
things: catalog metadata, a user count, and one throughput anchor. None of them
require transformers to load a checkpoint. So every quantization problem above —
ModelOpt, block-fp8, garbage generation — blocks the **Benchmark** tab and leaves
the TCO demo entirely untouched.

What *was* weak was the anchor. It came from:

1. the largest tok/s of the last HF benchmark run in the session, or
2. a hardcoded **1200 tok/s**,

and either way **one number was applied to every model**. That is why `$/MTok`
read identically across models differing 40x in active parameters — and 1200
tok/s is an aggregate-scale figure, wrong for single-stream decode on every
model in the catalog.

## Fix: per-model anchors from the vLLM captures

`vllm_reference.measured_tps(catalog_name)` returns the capture's n=1 per-user
figure — single-stream decode, which is the quantity the engine transfers across
hardware — plus a provenance string. The TCO tab prefers it, in this order:

1. **a vLLM capture for the selected model** — green, with precision and date;
2. **this session's HF benchmark** — amber, stating plainly that it is a
   *different* model's number and that capacity is unaffected;
3. **nothing** — red, saying capacity/nodes/CapEx are still sound (they come from
   attention geometry and measured checkpoint size) while the tok/s-derived
   figures are only as good as the number entered.

The silent 1200 fallback is gone; it now falls back to the slowest figure
actually measured (13.1), which errs toward under-claiming.

## Effect

Three models have real anchors, and the cost column finally separates them.
32 users, 8192 context, shared-weights, on one GB10:

| model | anchor | nodes | sessions/node | aggregate | $/MTok |
|---|---|---|---|---|---|
| Qwen2.5-7B | 13.1 tok/s | 1 | 226 | 296 tok/s | $0.3348 |
| Qwen3-8B-NVFP4 | 38.8 tok/s | 1 | 96 | 878 tok/s | $0.1130 |
| Qwen3.6-35B-A3B | 50.8 tok/s | 1 | **505** | **1,149 tok/s** | **$0.0863** |

The 35B MoE is simultaneously the cheapest per token and the highest
concurrency — the sparse-model story, now visible in cost rather than only in
capacity. Before this change all three showed the same $/MTok.

## What is still needed

Only more anchors. `scripts/recapture_reference.sh` has the fixed KV guard; the
35B arms need a larger `--kv-cache-memory` (or fewer seats) so pre-allocation
does not saturate the pool, and the MTP and Nemotron-Lightning arms have not run.
Each additional capture is one row in `_CATALOG_TO_CAPTURE` and one more model
with a defensible TCO. **Nothing here depends on the HF benchmark path being
fixed**, which is the part transformers blocks.

---

# Catalog rebuilt around what actually works — 2026-11-05

A catalog where every entry runs is worth more than a modern one full of
refusals. Six of eleven LLM entries could not run through HF, so they are out of
the benchmarkable list (they remain in `MODEL_CATALOG` for projection, and the
35B is still first-class in TCO via its vLLM anchor).

## Removed from the benchmarkable list

| Model | Why |
|---|---|
| Gemma-4-26B-A4B-NVFP4 | ModelOpt — loads and emits garbage |
| Nemotron-3.5-Lightning-30B-A3B | ModelOpt — raises |
| Nemotron-3-Super-120B-A12B | ModelOpt — raises |
| Qwen3.8-27B-FP8 | block-fp8 — loads and emits garbage |
| Qwen3.6-35B-A3B-FP8 | block-fp8 — loads and emits garbage; **kept in TCO** (vLLM anchor) |
| Laguna-S-2.1-NVFP4 | 110 GB against a 100 GB usable window |
| Nemotron VL x2 (vlm list) | ModelOpt |

## Added — older, plain, and verified

All already on disk, so **0 GB added** (symlinks):

| Model | Source | Size | Verified |
|---|---|---|---|
| Qwen3-0.6B | gb10-agent-stack | 1.5 G | **loads + coherent** |
| Qwen3-8B (BF16) | gb10-agent-stack | 16 G | **loads + coherent** |
| Mistral-7B-v0.1 | HF cache | 14 G | complete, 2/2 shards |

`Qwen3-8B` at BF16 is the most useful of the three: it pairs with
`Qwen3-8B-NVFP4` so the FP4 story becomes a **same-model** comparison rather than
an assertion.

**Rejected: Qwen2.5-32B-Instruct.** Its HF-cache entry looked usable at 53 GB but
is an abandoned download — **8 of 17 shards, 16 incomplete markers**. Caught by
loading it, not by inspecting it. That also explains the odd 53 GB against ~66 GB
for a BF16 32B. The symlink was removed.

`mem_guard` gained explicit sizes for the additions, because `qwen3-0.6b` matched
no key in the suffix table and fell through to the 7.0B default — a 0.6B model
reported as 7B. Same hazard as Laguna, third occurrence.

## The resulting demo

**Benchmark tab: 8 LLM models, every one verified to load and generate** — a
ladder from 0.6B to 14B, with a BF16-vs-NVFP4 pair at 8B. Vision: 4 encoders.

**TCO tab: 9 models** = the 8 benchmarkable plus Qwen3.6-35B-A3B, which has no HF
path but does have a measured vLLM anchor. That union is the right rule: the TCO
engine never loads a model, so "can I benchmark it here" and "can I cost it
credibly" are different questions.

## Anchors: two sources, vLLM preferred

`scripts/measure_hf_anchors.py` measures single-stream tok/s for each
HF-runnable model and writes `webui/data/hf_anchors.json`, with a coherence
fraction and a sample of the output so a silent-garbage model cannot become an
anchor. `best_anchor()` prefers a vLLM capture when one exists (that is the stack
the box serves with, and HF eager is slower), else the HF figure, and always
returns a provenance string the UI displays. Without this, every unmeasured
model fell back to one number and their $/MTok columns were identical — a
missing measurement masquerading as a result.

## Measured anchors, all eight — 2026-11-05

Every HF-runnable model measured once, single-stream, with an output sample and
an ASCII-coherence fraction so a silent-garbage model cannot become an anchor.
**All eight returned ascii 1.00 and readable prose.**

| model | HF eager | vLLM | anchor used |
|---|---|---|---|
| Qwen3-0.6B | **79.0** | — | HF 79.0 |
| Llama-3.2-3B | 25.6 | — | HF 25.6 |
| Mistral-7B-v0.1 | 14.5 | — | HF 14.5 |
| Qwen2.5-7B | 12.5 | **13.1** | vLLM 13.1 |
| Qwen3-8B | 11.3 | — | HF 11.3 |
| Qwen3-8B-NVFP4 | 6.0 | **38.8** | vLLM 38.8 |
| Phi-4 (14B) | 7.9 | — | HF 7.9 |
| Qwen2.5-14B | 7.7 | — | HF 7.7 |
| Qwen3.6-35B-A3B | (cannot load) | **50.8** | vLLM 50.8 |

A clean inverse-size curve — 79.0 at 0.6B down to 7.7 at 14B — which is what
bandwidth-bound decode should look like and is itself a demo point.

**The best single number in the table: Qwen3-8B-NVFP4 measures 6.0 tok/s through
HF and 38.8 through vLLM — 6.5x on identical weights and hardware.** That is the
serving-stack argument as a measurement rather than a claim, and it is why
`best_anchor()` prefers vLLM.

## The TCO demo, finished

9 models, every one with a measured anchor. 64 users, 8192 context,
shared-weights, one Dell Pro Max GB10:

| model | anchor | src | nodes | sessions/node | aggregate | $/MTok |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 79.0 | HF | 1 | 126 | 3,335 | **$0.0298** |
| Qwen3.6-35B-A3B | 50.8 | vLLM | 1 | **505** | 2,145 | $0.0463 |
| Qwen3-8B-NVFP4 | 38.8 | vLLM | 1 | 96 | 1,638 | $0.0606 |
| Llama-3.2-3B | 25.6 | HF | 1 | 121 | 1,082 | $0.0917 |
| Mistral-7B-v0.1 | 14.5 | HF | 1 | 98 | 613 | $0.1619 |
| Qwen2.5-7B | 13.1 | vLLM | 1 | 226 | 553 | $0.1794 |
| Qwen3-8B | 11.3 | HF | 1 | 86 | 478 | $0.2078 |
| Phi-4 (14B) | 7.9 | HF | **2** | 54 | 339 | $0.6249 |
| Qwen2.5-14B | 7.7 | HF | **2** | 57 | 328 | $0.6463 |

$/MTok now spans **22x** across the catalog, driven by per-model measurements
rather than one shared number. Change the user count and nodes, CapEx, aggregate
throughput and $/MTok all move off these anchors. A 35B MoE needing **one** node
while a 14B dense needs **two** is the sparse-model argument in a single row.

---

# Only ONE quantization sub-format works on this stack — 2026-11-05

Earlier I concluded "compressed-tensors works". That was too broad. Loading AND
generating from each sub-format, every time with a real generation rather than a
successful load:

| `format` | quant_method | example | resident | output |
|---|---|---|---|---|
| **`nvfp4-pack-quantized`** | compressed-tensors | nvidia/Qwen3-8B-NVFP4 | **5.96 GiB** (packed) | **coherent** |
| `pack-quantized` (int4/w4a16) | compressed-tensors | RedHatAI/Qwen3-30B-A3B-…w4a16 | 55.6 GiB | `\\",\\\\",\\\\"…` garbage |
| `pack-quantized` (int4/w4a16) | compressed-tensors | amd/Qwen3.6-35B-A3B-w4a16 | 64.6 GiB | `archyCalAMILrousciu…` garbage |
| `float-quantized` / block fp8 | fp8 | Qwen3.6-35B, Qwen3.8-27B | ~BF16 size | garbage |
| `modelopt` / `modelopt_mixed` | modelopt | Nemotrons, Gemma-4, Laguna | — | raises or garbage |
| (none) | plain BF16 | Qwen3-8B, Phi-4, … | 2 B/param | **coherent** |

So the working set is **plain BF16** and **`nvfp4-pack-quantized`** — nothing
else. Note the int4 w4a16 checkpoints also balloon to BF16 size on load (55-65
GiB from a 17-20 GB file), so they lose the "small loading weights" benefit as
well as being wrong.

## A real bug this exposed

`_is_prequantized_nvfp4()` chose the load path by matching the string **"NVFP4"
in the model name**. A compressed-tensors checkpoint named `*-w4a16` therefore
fell through to the bitsandbytes branch:

    The model is quantized with CompressedTensorsConfig but you are passing a
    BitsAndBytesConfig config.

Routing by filename was always going to break on the first vendor that named a
format differently, and it did. It now reads the declared `quant_method`, and
`checkpoint_quant_method()` accepts a direct path as well as an HF id.

Fixing the routing did NOT make those models work — it moved them from "wrong
error" to "correct path, still garbage" — which is the useful outcome: it
isolates the failure to the stack's int4 kernels rather than to our plumbing.

## Why Nemotron and Laguna work in their own projects

Different inference engine, not a broken model:

| project | engine | model |
|---|---|---|
| gb10-nemotron | `vllm/vllm-openai:v0.27.1` | Nemotron-3.5-Lightning-NVFP4 |
| gb10-laguna | `vllm/vllm-openai:v0.27.1` | Laguna-S-2.1-NVFP4 |
| gb10-markets | `vllm/vllm-openai:v0.27.1` | Qwen3.6-35B-A3B-FP8 |
| **gb10-demo Benchmark** | **HF transformers in-process** | its own catalog |

vLLM implements ModelOpt NVFP4 and block-FP8 kernels for sm_121; transformers
5.5.4 registers no ModelOpt quantizer at all and its fp8/int4 paths produce
garbage here. This is by design on the demo side — `vllm_reference.py:4` records
that the demo never runs vLLM in-process, so it cannot collide with the serving
projects.

Laguna has a second, independent blocker: 92.9 GiB needing
`--gpu-memory-utilization 0.92`, against the demo's 100 GB usable window. It
currently fails to start in its own project too.

The measurement that makes this a feature rather than an excuse:
**Qwen3-8B-NVFP4 is 6.0 tok/s under HF and 38.8 under vLLM — 6.5x on identical
weights and hardware.**

## 2026-11-06 — ModelOpt→compressed-tensors conversion does NOT generalize to Gemma-4

`scripts/convert_nvfp4_to_ct.py` is the reason `nvidia/Qwen3-8B-NVFP4` works: it
rewrote that ModelOpt checkpoint into `nvfp4-pack-quantized`, and
`nvidia--Qwen3-8B-NVFP4.modelopt-orig` is the pre-conversion backup. The script
was generalized today (`--src`/`--dst`, an abort when a checkpoint mixes FP8 into
NVFP4, and the processor/chat-template files a VLM needs) and tried on
`nvidia/Gemma-4-26B-A4B-NVFP4` — a promising target because
`Gemma4ForConditionalGeneration` **is** in transformers 5.5.4 and the checkpoint
carries a `vision_config`, so it would have been both an FP4 MoE anchor and the
VLM headline in one model.

**It failed, and the reason is not the quantization format.** The conversion runs
(18 s, 11,520 U8-packed modules, all with a `weight_scale_2`) and produces a
checkpoint whose fingerprint is `(compressed-tensors, nvfp4-pack-quantized)` — so
the new format guard *allows* it. It then loads in 4.5 s and generates **empty
output**. The load log lists hundreds of MISSING tensors, newly initialized:

    model.language_model.layers.{0...29}.mlp.up_proj.weight_global_scale      MISSING
    model.vision_tower.encoder.layers.{0...26}.self_attn.q_proj.linear.weight_packed  MISSING
    model.embed_vision.embedding_projection.weight_packed                     MISSING

The source and converted checkpoints have **identical** tensor names
(`model.language_model.layers.0.self_attn.q_proj.weight`,
`model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight`), which is
the clue: the converter only rewrites the quantization *suffixes*
(`weight`→`weight_packed`, `weight_scale_2`→`weight_global_scale`). For
`Qwen3ForCausalLM` that is sufficient. For this nested multimodal arch,
transformers' compressed-tensors integration expects the packed names at module
paths the ModelOpt export does not use — note the `.linear.` infix on every
vision-tower projection. A suffix rename cannot produce those; the converter
would need a per-architecture module-path remap.

Conclusion: the ModelOpt blocker is **per-architecture**, not per-format, so
"convert it" is not a general escape hatch. Also note the failed artifact was a
live hazard and was deleted: a converted-but-broken directory passes the format
guard, because the guard reads the declared format and the format was genuinely
correct. The checkpoint was wrong, not its label — the same
verified-metadata-is-not-verified-behaviour trap as every other finding here, now
reachable *through* the guard rather than around it.

Not retried: the two Nemotron VLMs. They are better prospects than they looked —
they ship their own modeling code with `auto_map` naming
`AutoModelForImageTextToText`, so they are not arch-blocked by stock transformers
— but `NemotronH_Omni_Reasoning_V3` is a custom class whose packed-weight
expectations are unknown, and the Super-VL is `modelopt_mixed` with almost no
quantized tensors in its first shard. Attempt only with a load+generate check as
the acceptance test, never a successful load.

## 2026-11-06 — both Nemotron VLMs: still unusable, and one of them took the box down

Attempted with load+generate as the acceptance test, per the rule that a successful
load is not evidence. Neither reached generation. The two have **different**
blockers, and neither is the one the audit predicted.

### Nemotron-3.5-Bolt-74B-A9B-24GB — two stacked blockers

Its `quantization_config` is not NVFP4 at all, despite the family naming:

    quant_algo: "Q8_0"   packing: "ggml"   group_size: 32
    effective_bits: 8.5  block_payload_bytes: 34     (3 config groups)

That is **GGUF/llama.cpp-style Q8_0 blocking in safetensors clothing** — 32 bytes of
payload plus a 2-byte scale per 34-byte block, which is why its shards carry U8
`weight` tensors with *no* accompanying scale tensors (the scales live inside the
block payload). `grep -ril "modelopt|nvfp4|dequant|quantiz" *.py` over its twelve
bundled modeling files returns **nothing**, so the remote code does not dequantize
either. transformers has no ggml path in `from_pretrained`.

It never gets that far regardless:

    ImportError: mamba-ssm is required by the Mamba model but cannot be imported

`mamba-ssm` publishes **no aarch64 wheel** — 2.3.2.post1 is a source tarball
requiring a CUDA compile against sm_121. The processor builds fine
(`NemotronH_Omni_Reasoning_V3Processor`); the model does not.

### Nemotron-3.5-Super-VL-120B — caused a global OOM

This one gets *past* Mamba (it falls back to the naive SSM implementation rather
than raising), and then:

    Unknown quantization type, got modelopt_mixed - ... Hence, we will skip the
    quantization.

That is the exact garbage signature this document records for Gemma-4. But the
attempt did something worse than produce garbage, and this is the part worth
keeping:

**A skipped quantization widens the memory footprint far beyond the file size.**
The checkpoint is 75.1 GB on disk. With quantization skipped, the load reached
**146 GB total-vm** and triggered a kernel OOM-kill that reaped the loading
process, **`systemd --user` for uid 1000**, `gb10-switch.service`, `pipewire`, the
Streamlit demo and an in-flight model download. The box did not need a power reset
this time, but losing the user systemd manager means **no user unit can be started
until root restarts `user@1000.service`** — there is no passwordless sudo here, so
:8000 had to be restored by running `switch_server.py` directly, unsupervised.

**Consequences for the guards, written down because this was not obvious:**

1. `precision_compatible()` refusing an unsupported `quant_method` **before loading**
   is a *memory-safety* property, not only a correctness one. The refusal is what
   stands between this box and a global OOM. Do not bypass it in a scratch script
   to "just see what happens" — that is exactly what happened here.
2. `mem_guard._MEASURED_GB` (on-disk size) is **not** a valid footprint proxy for a
   checkpoint whose quantization will be skipped. File size is a *lower* bound in
   that case, and the real figure was 1.9x it. Measured size is only meaningful for
   a checkpoint that will load *as* quantized.
3. The one-model rule was not violated — nothing else held the GPU — and the guard
   still would have refused this model through the UI. The failure was entirely in
   going around it.

### Status of the vision track

Unchanged: **no runnable VLM.** Both real VLMs on disk are blocked, Gemma-4's
conversion fails on module-path naming (see the entry above), and all six configured
vision entries are encoders. `benchmark_vlm()` stays implemented and unreachable.
The honest options are vLLM (which serves these models correctly) or
gb10-markets' Qwen3.6-35B, which is natively a VLM with its tower disabled by one
flag — not the HF benchmark path.
