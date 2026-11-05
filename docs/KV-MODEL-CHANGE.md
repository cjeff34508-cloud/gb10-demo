# KV-cache model change — 2026-11-05

Why 47 of the 90 golden-baseline rows moved, and why the move is a correction.

## What changed

`session_footprint()` and `model_memory_gb()` used to derive KV-per-session from
**total weight bytes**:

```python
kv_frac = min(_KV_FRAC_PER_TOKEN * context_len * batch_size, _KV_FRAC_CAP)
kv_per_session = raw * kv_frac          # raw = params_b * bytes_per_param
```

They now derive it from **attention geometry**, which is what actually sets KV:

```
bytes/token = 2 (K and V) x layers x kv_heads x head_dim x bytes(kv_dtype)
MLA         = layers x kv_latent x bytes(kv_dtype)
```

`weights_once` still comes from **total** parameters — an MoE holds every expert
resident even though it reads only a few per token. Only the KV term moved.

Models with no `MODEL_ARCH` entry fall back to the legacy estimate. Exactly one
LLM catalog entry is in that state today (`Nemotron-3-Super-120B`).

## Validation

The formula reproduces the only KV measurement taken on this box.
`Qwen3-Coder-30B-A3B-Instruct` is 48 layers, 4 kv_heads, head_dim 128, served
with `--kv-cache-dtype fp8`:

```
2 x 48 x 4 x 128 x 1 byte = 49,152 B = 48.0 KiB/token
```

qwen-hermes recorded "predicted 48.0 KB/token == observed". Exact match.

A trap worth recording: that model is `hidden_size 2048` with
`num_attention_heads 32`, so deriving `head_dim = hidden / q_heads` gives 64 —
**half** the real 128. `MODEL_ARCH` therefore carries an explicit optional
`head_dim` which takes precedence, and derivation is only the fallback.

## Why the old model was wrong

KV depends on layers, kv_heads and head_dim. Total parameter count is a proxy
that happens to track those for dense models of conventional shape, and breaks
in two directions elsewhere.

| Model | total B | active B | KV/token | KV @131K was | KV @131K now | old error |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 1.1 | — | 22.0 KiB | 1.0 GB | 3.0 GB | **3.0x under** |
| Qwen2.5-7B | 7.0 | — | 56.0 KiB | 6.1 GB | 7.5 GB | 1.2x under |
| Llama-3.3-70B | 70.0 | — | 320.0 KiB | 60.6 GB | 42.9 GB | 1.4x over |
| Mixtral-8x7B | 46.7 | 12.9 | 128.0 KiB | 40.4 GB | 17.2 GB | 2.4x over |
| Llama-3.1-405B | 405.0 | — | 504.0 KiB | 350.4 GB | 67.6 GB | 5.2x over |
| Qwen3-235B-A22B | 235.0 | 22.0 | 94.0 KiB | 203.3 GB | 12.6 GB | **16.1x over** |
| DeepSeek-V3 (671B) | 671.0 | 37.0 | 61.0 KiB | 290.2 GB | 8.2 GB | **35.4x over** |

The sparse models are where it collapses. `Qwen3-235B-A22B` has 94 KiB/token of
KV — *less than Llama-3.3-70B's 320 KiB* — because its 94 layers carry only 4
kv_heads. The old model charged it 203 GB because it has a lot of experts, and
experts are not KV. DeepSeek-V3 is worse still: MLA compresses KV to a 512-wide
latent per layer, which is the entire point of the design, and the old model
made it the most expensive thing in the catalog.

Small models went the other way: TinyLlama's weights are ~2 GB, so a
weights-relative KV estimate under-charged its real 22 KiB/token by 3x.

## Consequences in the baseline

47 of 90 rows changed; 43 are identical. The two that change a recommendation:

1. **Llama-3.1-405B @ 131K on XE9680 (H200 x8): 2 nodes -> 1 node.**
   CapEx `$1,065,180 -> $497,950`; $/MTok `0.80 -> 0.36`. The old model
   over-provisioned by $567,230 because it thought the model needed 1229 GB
   when it needs 945 GB.
2. **Mixtral-8x7B @ 131K on the Dell Pro Max GB10: "Cannot fit" -> 1 node.**
   The demo was telling customers that a 47B MoE will not run at long context
   on the box the demo runs on. It will.

Everything else is a $/MTok or tok/s shift inside an unchanged recommendation.

## Guard

`scripts/smoke_tco_capacity.py` was re-armed **before** this change (commit
`cd4e43c`), because assertion 1 pins replicated output to the baseline within
1e-6 and would have gone red by construction, telling us nothing.

Four relation-based assertions were added. Two were red before this change and
green after — they are regression tests, not decoration:

| | assertion | before | after |
|---|---|---|---|
| a8 | KV independent of expert count at equal attention geometry | **FAIL** 5.00x | PASS 1.00x |
| a11 | MLA KV below GQA KV at equal depth | **FAIL** 72.6 vs 5.1 GB | PASS 1.07 vs 4.30 GB |
| a9 | KV linear in context, zero at zero context | PASS | PASS |
| a10 | `weights_once` scales with total params, never `active_b` | PASS | PASS |

The pre-change baseline is preserved at `scripts/tco_golden_baseline.pre-kv.json`
so this table can be regenerated at any time.

## Not included in this change

The decode-rate side. Bytes read per token for an MoE track **active**
parameters, which is why a 35B-A3B feels fast at 273 GB/s, but
`calculate_tco()` still treats `gb10_tps` as a measured input belonging to the
target model and only transfers it across hardware by bandwidth ratio. Making
that model-relative needs a measured anchor on the sparse side, which is what
the vLLM re-capture adds. Deliberately separate so the two effects stay
attributable.
