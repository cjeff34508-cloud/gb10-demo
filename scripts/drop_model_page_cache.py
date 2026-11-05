#!/usr/bin/env python3
"""Drop the page cache for on-disk model weights.

posix_fadvise(POSIX_FADV_DONTNEED) asks the kernel to release the cached pages
for a file. It only evicts CLEAN pages — read-only weight files — so nothing is
written, nothing is lost, and no root is needed. The cache simply repopulates
the next time a model is read, which llm_inference._warm_file_cache() does on
purpose before every benchmark anyway.

Why this is needed: on this unified-memory box torch.cuda.mem_get_info() reports
MemFree, not MemAvailable, so ~100 GB of reclaimable page cache left over from a
large download looks like "no memory" to the benchmark's live-free guard.
"""
import os, sys
from pathlib import Path

EXT = {".safetensors", ".bin", ".pt", ".gguf"}
_DEFAULT_ROOTS = [
    Path.home() / "gb10-demo" / "models",
    # the symlinked checkpoints live in sibling projects + the HF cache
    Path.home() / "gb10-markets" / "models" / "hf" / "hub",
    Path.home() / "gb10-nemotron" / "models" / "hf" / "hub",
    Path.home() / "gb10-laguna" / "models" / "hf" / "hub",
    Path.home() / ".cache" / "huggingface" / "hub",
]
roots = [Path(p) for p in sys.argv[1:]] or _DEFAULT_ROOTS

def meminfo(k):
    for line in open("/proc/meminfo"):
        if line.startswith(k):
            return int(line.split()[1]) / 1048576
    return 0.0

before_cached, before_free = meminfo("Cached"), meminfo("MemFree")
n = dropped = 0
for root in roots:
    for f in root.rglob("*"):
        try:
            if not f.is_file() or f.suffix not in EXT:
                continue
            sz = f.stat().st_size
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, sz, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            n += 1; dropped += sz
        except (OSError, PermissionError):
            continue

after_cached, after_free = meminfo("Cached"), meminfo("MemFree")
print(f"  advised {n} weight files, {dropped/1e9:.1f} GB of file data")
print(f"  Cached   {before_cached:7.1f} -> {after_cached:7.1f} GiB  ({before_cached-after_cached:+.1f})")
print(f"  MemFree  {before_free:7.1f} -> {after_free:7.1f} GiB  ({after_free-before_free:+.1f})")
