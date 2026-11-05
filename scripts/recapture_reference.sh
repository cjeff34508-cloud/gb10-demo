#!/usr/bin/env bash
# C5 — re-capture the vLLM serving reference, with MoE anchors.
#
# WHY: webui/data/vllm_reference.json held only two captures, both <=8B DENSE,
# from 2026-07-21, one with a non-monotonic point taken at kv=98%. The TCO
# engine calibrates _BATCH_BW_SHARE_EXP against that file and transfers a single
# measured tok/s to every model, so $/MTok is currently identical across models
# that differ 40x in active parameters. MoE anchors are what fix that.
#
# CAPTURE CONFIG IS NOT THE PRODUCTION CONFIG, deliberately:
#   markets-vllm serves the 35B at --max-model-len 262144 --max-num-seqs 4,
#   tuned for long-context single-user work. A concurrency sweep needs seats, so
#   these arms use a short context and 32 seats. That measures how aggregate
#   throughput SCALES WITH CONCURRENCY — the quantity the exponent models —
#   rather than the as-deployed long-context behaviour. Both are real; they
#   answer different questions, and the file records which one this is.
#
# One model at a time, through one-model-guard.sh. Takes the GPU for ~45 min.
set -uo pipefail
DEMO="$HOME/gb10-demo"
PY="$DEMO/env/llm/bin/python"
CAP="$DEMO/scripts/capture_vllm_reference.py"
IMAGE="vllm/vllm-openai:v0.27.1"
LOG="$DEMO/logs/recapture-$(date +%Y%m%d-%H%M%S).log"
PORT=8599                      # not a project port — avoids colliding with any unit
NAME="c5-capture"
CONC="1,2,4,8,16,32"

mkdir -p "$DEMO/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== C5 recapture $(date -Is) ==="

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

serve() {  # serve <mode:dir|hub> <host-path> <model-ref> <served-name> <extra-flags...>
  local mode="$1" hostpath="$2" modelref="$3" served="$4"; shift 4
  cleanup
  bash "$HOME/gb10-switch/one-model-guard.sh" "$NAME" || { echo "✗ guard refused"; return 75; }
  echo "→ launching vLLM: $served  ($mode)"

  # HF-cache snapshots are directories of SYMLINKS into ../../blobs/, so mounting
  # the snapshot alone leaves every weight dangling inside the container (that is
  # what killed the first run: 56-69 broken links). For those, mount the hub ROOT
  # and address the model by repo id via HF_HOME — exactly what markets-vllm does.
  local -a mnt
  if [ "$mode" = "hub" ]; then
    mnt=(-v "$hostpath:/models/hf" -e HF_HOME=/models/hf)
  else
    mnt=(-v "$hostpath:/model")
  fi

  # NOTE: no --rm. An exited --rm container is deleted before its logs can be
  # read, which is why the first run reported "No such container" instead of the
  # actual error. cleanup() removes it explicitly.
  docker run -d --name "$NAME" --gpus all --ipc=host \
    -p 127.0.0.1:${PORT}:${PORT} \
    "${mnt[@]}" \
    -e HF_HUB_OFFLINE=1 -e VLLM_USE_DEEP_GEMM=0 \
    -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
    "$IMAGE" \
    --model "$modelref" --served-model-name "$served" --port "$PORT" \
    --max-model-len 16384 --max-num-seqs 32 \
    --gpu-memory-utilization 0.55 \
    --kv-cache-memory=8589934592 --kv-cache-dtype fp8 \
    "$@" >/dev/null || { echo "  ✗ docker run failed"; return 1; }

  for _ in $(seq 1 300); do   # 25 min ceiling
    curl -sf -m 3 -H 'Authorization: Bearer local-vllm' \
      "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "  ✓ serving"; return 0; }
    if ! docker ps -q -f name="$NAME" -f status=running | grep -q .; then
      echo "  ✗ container exited — last 30 log lines:"
      docker logs --tail 30 "$NAME" 2>&1 | sed 's/^/      /'
      return 1
    fi
    sleep 5
  done
  echo "  ✗ never became ready — last 30 log lines:"
  docker logs --tail 30 "$NAME" 2>&1 | sed 's/^/      /'
  return 1
}

capture() {  # capture <key> <precision> <served-name> [extra capture args...]
  local key="$1" prec="$2" served="$3"; shift 3
  echo "→ capturing $key"
  "$PY" "$CAP" --base-url "http://127.0.0.1:${PORT}/v1" --model "$served" \
    --key "$key" --precision "$prec" --concurrency "$CONC" --kv-max 0.85 "$@" \
    || echo "  ✗ capture failed for $key"
  cleanup; sleep 10
}

M="$DEMO/models/llm-models"
ARMS_RUN=0

# 1. The existing NVFP4 anchor, re-captured with the KV ceiling honoured.
#    MODE=completions: this checkpoint ships NO chat template, so
#    /v1/chat/completions 400s. That is what failed the first run — and the
#    capture script then wrongly recorded fp4_supported=false over a good
#    capture, which is now narrowed to real FP4 failures only.
if serve dir "$M/nvidia--Qwen3-8B-NVFP4" /model qwen3-8b-nvfp4; then
  capture qwen3-8b-nvfp4 NVFP4 qwen3-8b-nvfp4 --mode completions --fp4
  ARMS_RUN=$((ARMS_RUN+1))
fi

# 2. First MoE anchor, WITHOUT MTP — the baseline for the spec-decode delta.
if serve hub "$HOME/gb10-markets/models/hf" Qwen/Qwen3.6-35B-A3B-FP8 qwen3.6-35b-a3b \
         --moe-backend triton; then
  capture qwen3.6-35b-a3b-fp8 FP8 qwen3.6-35b-a3b; ARMS_RUN=$((ARMS_RUN+1))
fi

# 3. Same model WITH MTP speculative decoding — the pair the TCO engine needs
#    before spec-decode can be a term rather than an omission.
if serve hub "$HOME/gb10-markets/models/hf" Qwen/Qwen3.6-35B-A3B-FP8 qwen3.6-35b-a3b \
         --moe-backend triton \
         --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}'; then
  capture qwen3.6-35b-a3b-fp8-mtp FP8 qwen3.6-35b-a3b; ARMS_RUN=$((ARMS_RUN+1))
fi

# 4. Second MoE anchor: 30B/3B NVFP4 — and the only way to benchmark a ModelOpt
#    checkpoint at all, since transformers cannot load one (docs/CATALOG-2026.md).
if serve hub "$HOME/gb10-nemotron/models/hf" \
         nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 nemotron-3.5-lightning \
         --moe-backend marlin; then
  capture nemotron-3.5-lightning-30b-a3b NVFP4 nemotron-3.5-lightning --fp4
  ARMS_RUN=$((ARMS_RUN+1))
fi

echo "=== done: $ARMS_RUN/4 arms captured $(date -Is) ==="
"$PY" - <<'PYEOF'
import json
from pathlib import Path
p = Path.home()/"gb10-demo"/"webui"/"data"/"vllm_reference.json"
d = json.loads(p.read_text())
print(f"\ncaptures now in {p.name}:")
for k, c in (d.get("captures") or {}).items():
    rows = c.get("per_n") or []
    top = rows[-1] if rows else {}
    print(f"  {k:34s} {c.get('precision','?'):6s} levels={len(rows):2d} "
          f"peak_agg={top.get('aggregate_tps',0):8.1f} tok/s kv={top.get('kv_cache_pct',0)*100:.0f}%")
PYEOF
