#!/usr/bin/env python3
"""
capture_vllm_reference.py — capture a REAL vLLM batched-serving reference for the demo.

The GB10 demo (in-process HuggingFace eager) and the vLLM Docker container cannot co-run
(ONE heavy GPU workload at a time). So this is an OFFLINE capture tool: the operator brings
up the vLLM container, runs this once, and it writes webui/data/vllm_reference.json. The
Streamlit app only *reads* that file — it never talks to vLLM live.

It fires N concurrent /v1/chat/completions requests at each concurrency level, measures real
per-user and aggregate tok/s, and samples vLLM's Prometheus /metrics (KV-cache usage, running
requests) plus host GPU util — the numbers that show the true production delta vs the demo's
HF-eager runs.

Typical use
-----------
  # 1) Same-model production delta — serve a model that is ALSO in the demo's HF catalog,
  #    then capture (pass the HF-eager single-user tok/s you measured in the demo for the delta):
  #    (serve e.g. Qwen2.5-7B-Instruct via a serve_qwen*-style command on :8503 first)
  python scripts/capture_vllm_reference.py --key qwen2.5-7b --precision FP8 \
      --hf-eager-tps 44

  # 2) Experimental REAL FP4 — serve the NVFP4 checkpoint via vLLM, then capture with --fp4:
  #    docker ... vllm serve /models/.../nvidia--Qwen3-8B-NVFP4  (on :8503)
  python scripts/capture_vllm_reference.py --key qwen3-8b-nvfp4 --precision NVFP4 --fp4

If the container is down, the script aborts with guidance (it never launches vLLM itself).
With --fp4, a serving/inference failure is recorded as {fp4_supported: false, error: ...} so
the demo falls back to the honest 'FP4 -> BF16 compute' relabel.
"""

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_WEBUI = Path(__file__).resolve().parent.parent / "webui"
sys.path.insert(0, str(_WEBUI))
try:
    from helpers.bottleneck_analyzer import get_gpu_stats
except Exception:
    def get_gpu_stats():
        return {}

REF_PATH_DEFAULT = _WEBUI / "data" / "vllm_reference.json"
_AUTH = {"Authorization": "Bearer local-vllm"}


# --------------------------------------------------------------------------- HTTP helpers
def _urls(base_url: str):
    base = base_url.rstrip("/")
    root = base[:-3].rstrip("/") if base.endswith("/v1") else base
    return {
        "models":      base + "/models",
        "chat":        base + "/chat/completions",
        "completions": base + "/completions",
        "metrics":     root + "/metrics",
    }


def _http_get(url: str, timeout: float = 10.0) -> str:
    req = urllib.request.Request(url, headers=_AUTH)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def _parse_prom(text: str, name: str):
    """First sample value for a Prometheus metric line (handles optional {labels})."""
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        try:
            return float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return None


def detect_model(urls) -> str:
    data = json.loads(_http_get(urls["models"]))
    return data["data"][0]["id"]


# --------------------------------------------------------------------------- load generation
async def _one_request(session, urls, model, prompt, max_tokens, mode):
    # "completions" uses the raw prompt endpoint — needed for checkpoints whose tokenizer
    # ships no chat template (e.g. the repackaged NVFP4 checkpoint), which /chat/completions
    # rejects. "chat" is the default for normal instruct models.
    if mode == "completions":
        url = urls["completions"]
        body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": 0.7, "top_p": 0.8}
    else:
        url = urls["chat"]
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0.7, "top_p": 0.8,
                "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    async with session.post(url, json=body, headers=_AUTH) as resp:
        data = await resp.json()
    dt = time.perf_counter() - t0
    if "choices" not in data:
        raise RuntimeError(f"vLLM error response: {str(data)[:200]}")
    toks = int((data.get("usage") or {}).get("completion_tokens", 0) or 0)
    return dt, toks


async def _sampler(metrics_url, stop_evt, out):
    """Poll /metrics + host GPU stats until stopped; keep peak KV / running / util."""
    while not stop_evt.is_set():
        try:
            txt = await asyncio.to_thread(_http_get, metrics_url, 5.0)
            kv = _parse_prom(txt, "vllm:kv_cache_usage_perc")
            run = _parse_prom(txt, "vllm:num_requests_running")
        except Exception:
            kv = run = None
        gs = get_gpu_stats()
        if kv is not None:
            out["kv"] = max(out["kv"], kv * 100.0 if kv <= 1.0 else kv)
        if run is not None:
            out["run"] = max(out["run"], run)
        _u = gs.get("gpu_util_pct")  # pynvml util is unavailable on this GB10 → stays None
        if _u is not None:
            out["gpu_util"] = _u if out["gpu_util"] is None else max(out["gpu_util"], _u)
        await asyncio.sleep(0.25)


async def run_level(session, urls, model, prompt, max_tokens, n_users, mode):
    peak = {"kv": 0.0, "run": 0.0, "gpu_util": None}
    stop_evt = asyncio.Event()
    sampler = asyncio.create_task(_sampler(urls["metrics"], stop_evt, peak))

    t_start = time.perf_counter()
    results = await asyncio.gather(
        *[_one_request(session, urls, model, prompt, max_tokens, mode) for _ in range(n_users)],
        return_exceptions=True,
    )
    wall = time.perf_counter() - t_start
    stop_evt.set()
    await sampler

    errors = [r for r in results if isinstance(r, Exception)]
    ok = [r for r in results if not isinstance(r, Exception)]
    if not ok:
        raise RuntimeError(f"all {n_users} requests failed; first error: {errors[0]}")

    per_user_tps_vals = [toks / dt for dt, toks in ok if dt > 0]
    total_toks = sum(toks for _, toks in ok)
    return {
        "n_users": n_users,
        "per_user_tps": round(sum(per_user_tps_vals) / len(per_user_tps_vals), 1),
        "aggregate_tps": round(total_toks / wall, 1) if wall > 0 else 0.0,
        "kv_cache_pct": round(peak["kv"], 2),
        "gpu_util": round(peak["gpu_util"], 1) if peak["gpu_util"] is not None else None,
        "running_peak": int(peak["run"]),
        "completed": len(ok),
        "failed": len(errors),
    }


async def capture(args, urls, model):
    try:
        import aiohttp
    except ImportError:
        sys.exit("✗ aiohttp not available in this venv — run inside env/llm.")

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    per_n = []
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    conn = aiohttp.TCPConnector(limit=max(levels) + 4)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        for n in levels:
            print(f"  → concurrency {n} …", flush=True)
            row = await run_level(session, urls, model, args.prompt, args.max_tokens, n, args.mode)
            _u = row["gpu_util"]
            print(f"    per-user {row['per_user_tps']:.0f} tok/s · "
                  f"aggregate {row['aggregate_tps']:.0f} tok/s · "
                  f"KV {row['kv_cache_pct'] * 100:.0f}% · peak-running {row['running_peak']} · "
                  f"util {(f'{_u:.0f}%' if _u is not None else 'N/A')} "
                  f"({row['completed']}/{n} ok)", flush=True)
            per_n.append(row)
            # vllm:kv_cache_usage_perc is a 0..1 fraction. Above ~0.85 the pool is
            # effectively full: vLLM starts preempting/queueing, so higher
            # concurrency measures the KV ceiling rather than how aggregate
            # throughput scales — which is the thing this reference calibrates.
            # The 2026-07 capture ran to kv=0.98 and produced a non-monotonic
            # point (n=8: 24.3 tok/s/user vs n=16: 37.1), which is why this stops.
            if args.kv_max and row["kv_cache_pct"] >= args.kv_max:
                print(f"    ⏹ stopping sweep: KV {row['kv_cache_pct']*100:.0f}% "
                      f"≥ {args.kv_max*100:.0f}% ceiling — higher concurrency would "
                      f"measure cache pressure, not scaling", flush=True)
                break
    return per_n


# --------------------------------------------------------------------------- persistence
def write_capture(out_path: Path, key: str, capture_block: dict):
    try:
        data = json.loads(out_path.read_text())
    except Exception:
        data = {}
    data.setdefault("captures", {})[key] = capture_block
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"✓ wrote capture '{key}' → {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8503/v1", help="vLLM OpenAI base URL")
    ap.add_argument("--model", default=None, help="served model id (default: auto-detect from /v1/models)")
    ap.add_argument("--key", required=True, help="capture key, e.g. qwen2.5-7b or qwen3-8b-nvfp4")
    ap.add_argument("--precision", default="FP8", help="precision label for this capture")
    ap.add_argument("--concurrency", default="1,2,4,8,16", help="comma-separated concurrency levels")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--mode", choices=["chat", "completions"], default="chat",
                    help="chat = /v1/chat/completions (default); completions = /v1/completions "
                         "(use for checkpoints whose tokenizer has no chat template, e.g. NVFP4)")
    ap.add_argument("--prompt", default="Write a Python function that returns the nth Fibonacci number, with a short docstring.")
    ap.add_argument("--hf-eager-tps", type=float, default=None,
                    help="matched single-user HF-eager tok/s from the demo (enables the production-delta card)")
    ap.add_argument("--fp4", action="store_true",
                    help="mark this as the experimental NVFP4 real-FP4 capture (records fp4_supported)")
    ap.add_argument("--kv-max", type=float, default=0.85,
                    help="stop escalating concurrency once KV-cache usage reaches this "
                         "fraction (0..1). 0 disables. Default 0.85 — beyond that the "
                         "sweep measures the cache ceiling, not throughput scaling.")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request timeout (s)")
    ap.add_argument("--gpu", default="GB10")
    ap.add_argument("--out", default=str(REF_PATH_DEFAULT))
    args = ap.parse_args()

    urls = _urls(args.base_url)

    # Health check — never launch vLLM ourselves (respects one-workload-at-a-time).
    try:
        model = args.model or detect_model(urls)
    except Exception as e:
        msg = (f"✗ vLLM not reachable at {urls['models']} ({e}).\n"
               f"  Bring the container up first (e.g. ~/qwen-hermes/scripts/stack-up.sh or a serve_*.sh),\n"
               f"  confirm: curl -s {urls['models']} | python3 -m json.tool")
        if args.fp4:
            # Record the failure so the demo keeps the honest BF16-compute relabel.
            write_capture(Path(args.out), args.key, {
                "model": args.model, "precision": args.precision, "gpu": args.gpu,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "per_n": [], "hf_eager_tps": args.hf_eager_tps,
                "fp4_supported": False, "error": str(e),
                "notes": "NVFP4 serving unreachable — demo falls back to honest FP4->BF16 relabel",
            })
        sys.exit(msg)

    print(f"✓ vLLM up — serving '{model}'  (key={args.key}, precision={args.precision})")
    block = {
        "model": model, "served_name": model, "precision": args.precision, "gpu": args.gpu,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hf_eager_tps": args.hf_eager_tps,
        "notes": f"vLLM batched serving, max_tokens={args.max_tokens}",
    }
    try:
        block["per_n"] = asyncio.run(capture(args, urls, model))
        if args.fp4:
            block["fp4_supported"] = True  # NVFP4 was served and produced real output
            block["notes"] += " · real NVFP4 tensor-core path (vLLM)"
    except Exception as e:
        if args.fp4:
            # Only blame FP4 for failures that could plausibly BE an FP4 problem.
            # A chat-template or transport error says nothing about tensor-core
            # support, and recording fp4_supported=false for one makes the demo
            # claim real FP4 does not work when it demonstrably does. Measured
            # 2026-11-05: nvidia/Qwen3-8B-NVFP4 ships no chat template, so
            # /v1/chat/completions 400s — and that overwrote a GOOD capture with
            # fp4_supported=false.
            _msg = str(e).lower()
            _not_fp4 = any(t in _msg for t in (
                "chat template", "chat_template", "badrequest", "404", "not found",
                "connection", "timeout", "unauthorized", "model not found",
            ))
            block["per_n"] = block.get("per_n", [])
            block["error"] = str(e)
            if _not_fp4:
                block["notes"] += (" · capture failed for a NON-FP4 reason (see error); "
                                   "fp4_supported left UNSET rather than claiming false")
                block.pop("fp4_supported", None)
                write_capture(Path(args.out), args.key, block)
                sys.exit(f"✗ capture failed — NOT attributed to FP4: {e}")
            block["fp4_supported"] = False
            block["notes"] += " · NVFP4 serving/inference failed — demo keeps FP4->BF16 relabel"
            write_capture(Path(args.out), args.key, block)
            sys.exit(f"✗ capture failed (recorded fp4_supported=false): {e}")
        sys.exit(f"✗ capture failed: {e}")

    write_capture(Path(args.out), args.key, block)


if __name__ == "__main__":
    main()
