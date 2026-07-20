#!/usr/bin/env python3
"""
Dell Pro Max GB10 Demo Suite Model Downloader
Downloads HuggingFace models for all workload modules.
Stores to NVMe-backed models directory. Idempotent: skips already-downloaded models.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Use hf_transfer (Rust-based) for parallel chunk downloads with proper timeouts.
# Prevents the xet-protocol stall that causes downloads to hang at ~60%.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "120")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed. Install with: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

# Configuration — all paths resolve to /dev/nvme0n1 (3.6 TB NVMe, system root)
PROJECT_DIR = Path.home() / "gb10-demo"
CONFIG_FILE = PROJECT_DIR / "config" / "model-lists.json"
LOG_FILE    = PROJECT_DIR / "logs" / "model-download.log"
MODELS_DIR  = PROJECT_DIR / "models"   # on nvme0n1p2 → /
HF_TOKEN_FILE = Path.home() / "hf-token"


def _nvme_device(path: Path) -> str:
    """Return the underlying block device for a given path (best-effort)."""
    try:
        out = subprocess.check_output(
            ["df", "--output=source", str(path)], text=True
        ).strip().splitlines()
        return out[-1] if len(out) > 1 else "unknown"
    except Exception:
        return "unknown"


def get_hf_token() -> str | None:
    """Read HF token from project token file or HF cache."""
    if HF_TOKEN_FILE.exists():
        return HF_TOKEN_FILE.read_text().strip()
    cache_token = Path.home() / ".cache" / "huggingface" / "token"
    if cache_token.exists():
        return cache_token.read_text().strip()
    return os.environ.get("HF_TOKEN")

# Logging setup
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_model_lists():
    """Load model lists from config file."""
    if not CONFIG_FILE.exists():
        logger.warning(f"Config file not found: {CONFIG_FILE}")
        logger.info("Creating default empty config...")
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        default_config = {
            "hpc": [],
            "llm": ["meta-llama/Llama-2-7b-hf"],
            "vlm": ["openai/clip-vit-base-patch32"],
            "cnn": ["google/vit-base-patch16-224"],
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(default_config, f, indent=2)
        logger.info(f"Default config created: {CONFIG_FILE}")
        return default_config

    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def model_exists(workload, model_id):
    """Check if model weights are already downloaded (not just metadata)."""
    model_path = MODELS_DIR / f"{workload}-models" / model_id.replace("/", "--")
    if not model_path.exists():
        return False
    weight_extensions = (".safetensors", ".bin", ".pt", ".ckpt", ".msgpack", ".h5")
    return any(
        f.suffix in weight_extensions
        for f in model_path.iterdir()
        if f.is_file() and not f.name.startswith(".")
    )


STALL_TIMEOUT = 600  # seconds with no new bytes before declaring a stall


def _dir_size(path: Path) -> int:
    """Sum of all file sizes under path (best-effort)."""
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except Exception:
        return 0


def _download_with_watchdog(workload, model_id, model_path, token):
    """Run snapshot_download in a thread; restart if no progress for STALL_TIMEOUT seconds."""
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        result = {"done": False, "error": None}
        last_size = [_dir_size(model_path)]
        last_progress_time = [time.monotonic()]
        stall_detected = threading.Event()

        def _do_download():
            try:
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(model_path.parent),
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    token=token,
                )
                result["done"] = True
            except Exception as e:
                result["error"] = e
            finally:
                stall_detected.set()

        def _watchdog():
            while not stall_detected.is_set():
                time.sleep(30)
                if stall_detected.is_set():
                    break
                current_size = _dir_size(model_path)
                if current_size > last_size[0]:
                    last_size[0] = current_size
                    last_progress_time[0] = time.monotonic()
                elif time.monotonic() - last_progress_time[0] > STALL_TIMEOUT:
                    logger.warning(
                        f"[{workload}] Stall detected for {model_id} "
                        f"(no progress for {STALL_TIMEOUT}s, attempt {attempt}/{max_attempts})"
                    )
                    stall_detected.set()

        dl_thread = threading.Thread(target=_do_download, daemon=True)
        wd_thread = threading.Thread(target=_watchdog, daemon=True)
        dl_thread.start()
        wd_thread.start()
        dl_thread.join()
        stall_detected.set()
        wd_thread.join()

        if result["done"]:
            return True
        if result["error"] is not None:
            raise result["error"]
        # stall — loop back and retry (snapshot_download resumes automatically)
        logger.info(f"[{workload}] Retrying {model_id} (attempt {attempt + 1}/{max_attempts})...")

    raise RuntimeError(f"Download of {model_id} stalled {max_attempts} times and did not complete")


def download_model(workload, model_id):
    """Download a single model from HuggingFace Hub."""
    model_path = MODELS_DIR / f"{workload}-models" / model_id.replace("/", "--")

    if model_exists(workload, model_id):
        logger.info(f"[{workload}] Model already exists: {model_id}")
        return True

    try:
        logger.info(f"[{workload}] Starting download: {model_id}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        token = get_hf_token()
        _download_with_watchdog(workload, model_id, model_path, token)
        logger.info(f"[{workload}] ✓ Downloaded: {model_id}")
        return True

    except Exception as e:
        logger.error(f"[{workload}] ✗ Failed to download {model_id}: {e}")
        return False


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Dell Pro Max GB10 Demo Suite Model Downloader started")
    _dev = _nvme_device(MODELS_DIR)
    try:
        _df = subprocess.check_output(
            ["df", "-h", "--output=avail,pcent", str(MODELS_DIR)], text=True
        ).strip().splitlines()
        _space = _df[-1].strip() if len(_df) > 1 else "unknown"
    except Exception:
        _space = "unknown"
    logger.info(f"Storage device : {_dev}")
    logger.info(f"Models dir     : {MODELS_DIR}")
    logger.info(f"Disk available : {_space}")
    logger.info("=" * 60)

    model_lists = load_model_lists()

    if not model_lists:
        logger.warning("No models to download")
        return

    # Flatten model list with workload metadata
    tasks = []
    for workload, models in model_lists.items():
        for model_id in models:
            if model_id.strip():  # Skip empty strings
                tasks.append((workload, model_id))

    if not tasks:
        logger.info("No models configured for download")
        return

    logger.info(f"Total models to download: {len(tasks)}")

    completed = 0
    failed = 0

    for workload, model_id in tasks:
        try:
            success = download_model(workload, model_id)
            if success:
                completed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Task error for {model_id}: {e}")
            failed += 1

    logger.info("=" * 60)
    logger.info(f"Model download complete: {completed} successful, {failed} failed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
