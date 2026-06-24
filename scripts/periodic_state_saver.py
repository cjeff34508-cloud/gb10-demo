#!/usr/bin/env python3
"""
Periodic state saver — saves session history + project progress checkpoints to NVMe.
Outputs JSON to stdout for the cron job to deliver.
"""

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Paths
SAVE_DIR = Path("/home/cjeff/gb10-demo/state_snapshots")
SESSION_DB = Path("/home/cjeff/.hermes/sessions.db")
PROJECT_DIR = Path("/home/cjeff/gb10-demo")

# Key project files to snapshot
KEY_FILES = [
    "webui/helpers/tco_engine.py",
]

def get_recent_sessions(limit=10):
    """Extract recent session metadata from the SQLite DB."""
    sessions = []
    if not SESSION_DB.exists():
        return sessions
    try:
        conn = sqlite3.connect(str(SESSION_DB))
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT session_id, title, when, source FROM sessions ORDER BY when DESC LIMIT ?",
            (limit,),
        )
        for row in cur.fetchall():
            sessions.append({
                "session_id": row["session_id"],
                "title": row["title"],
                "when": row["when"],
                "source": row["source"],
            })
        conn.close()
    except Exception as e:
        sessions.append({"error": str(e)})
    return sessions


def get_key_file_checksums():
    """SHA-256 checksums of key project files."""
    results = {}
    for rel in KEY_FILES:
        fpath = PROJECT_DIR / rel
        if fpath.exists():
            import hashlib
            h = hashlib.sha256()
            h.update(fpath.read_bytes())
            results[rel] = {
                "sha256": h.hexdigest()[:16],
                "size_bytes": fpath.stat().st_size,
                "modified": datetime.fromtimestamp(fpath.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        else:
            results[rel] = {"error": "not found"}
    return results


def get_project_file_count():
    """Total files and line counts in the project directory."""
    total_files = 0
    total_lines = 0
    try:
        for root, dirs, files in PROJECT_DIR.walk():
            # Skip .git and __pycache__
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".venv", "venv")]
            for f in files:
                if f.endswith((".py", ".md", ".json", ".yaml", ".yml", ".html", ".js", ".css")):
                    fpath = Path(root) / f
                    try:
                        total_files += 1
                        total_lines += sum(1 for _ in open(fpath, "r", errors="replace"))
                    except Exception:
                        pass
    except Exception as e:
        return {"error": str(e)}
    return {"total_files": total_files, "total_lines": total_lines}


def get_git_status():
    """Git status of the project directory."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "branch": subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip(),
            "modified_files": len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0,
            "last_commit": subprocess.run(
                ["git", "log", "-1", "--format=%H %s %ci"],
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip(),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot = {
        "timestamp": timestamp,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "recent_sessions": get_recent_sessions(limit=10),
        "key_file_checksums": get_key_file_checksums(),
        "project_stats": get_project_file_count(),
        "git_status": get_git_status(),
    }

    # Save to file
    out_path = SAVE_DIR / f"snapshot_{timestamp}.json"
    out_path.write_text(json.dumps(snapshot, indent=2))

    # Also update the latest symlink/alias
    latest_path = SAVE_DIR / "latest.json"
    latest_path.write_text(json.dumps(snapshot, indent=2))

    # Print summary to stdout
    print(json.dumps({
        "status": "ok",
        "snapshot": str(out_path),
        "sessions_saved": len(snapshot["recent_sessions"]),
        "project_files_tracked": snapshot["project_stats"].get("total_files", 0),
        "project_lines_tracked": snapshot["project_stats"].get("total_lines", 0),
        "git_modified_files": snapshot["git_status"].get("modified_files", 0),
        "last_commit": snapshot["git_status"].get("last_commit", "N/A"),
    }, indent=2))


if __name__ == "__main__":
    main()
