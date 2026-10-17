"""CPU "real queue" check for CI and for a clean-install test: start a real ComfyUI server, install this node pack into it, and run a
workflow of the model-free nodes through POST /prompt.

    python scripts/ci_queue_check.py --comfyui <ComfyUI checkout>

The scoring itself needs the trained weights (about 15 GB), so CI cannot run it; this check proves that the pack loads in a stock
ComfyUI, registers all its nodes and that `Load Video -> SyncReward Shift Audio -> Get Video Components` executes through the queue
and moves the audio. The real-weights runs are documented in docs/VERIFICATION.md. The server is started and stopped by this script only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import queue_client  # noqa: E402
from sample_video import make_sample_video  # noqa: E402


def wait_up(server: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{server}/system_stats", timeout=3).read()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("ComfyUI did not start in time")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8191)
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args(argv)

    comfy = args.comfyui.resolve()
    pack = comfy / "custom_nodes" / "ComfyUI-SyncReward"
    if pack.exists():
        shutil.rmtree(pack)
    shutil.copytree(ROOT, pack, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs"))
    (comfy / "input").mkdir(exist_ok=True)
    make_sample_video(comfy / "input" / "example.mp4")

    log_dir = args.workdir or Path(tempfile.mkdtemp(prefix="syncreward-ci-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / "comfyui.log").open("w", encoding="utf-8")
    server = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(args.port), "--disable-auto-launch", "--disable-api-nodes"],
        cwd=comfy, stdout=log, stderr=subprocess.STDOUT,
    )
    failures = []
    try:
        wait_up(server, proc)
        info = json.loads(urllib.request.urlopen(f"{server}/object_info", timeout=30).read())
        expected = {"SyncRewardLoader", "SyncRewardScore", "SyncRewardSelectBest", "SyncRewardShiftAudio", "SyncRewardUnload"}
        print("registered:", sorted(expected & set(info)))
        if expected - set(info):
            failures.append(f"nodes missing from /object_info: {sorted(expected - set(info))}")

        workflow = {
            "1": {"class_type": "LoadVideo", "inputs": {"file": "example.mp4"}},
            "2": {"class_type": "SyncRewardShiftAudio", "inputs": {"video": ["1", 0], "offset_ms": 250.0}},
            "3": {"class_type": "GetVideoComponents", "inputs": {"video": ["2", 0]}},
            "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 2]}},
        }
        result = queue_client.run(server, workflow, None, 300)
        print(json.dumps({"workflow": "shift_audio", "status": result["status"], "seconds": result["wall_seconds"], "error": result.get("error")}))
        if result["status"] != "success":
            failures.append(f"shift_audio workflow: {result.get('error') or result['status']}")
        elif abs(float(next(iter(result["text"].values()))[0]) - 25.0) > 0.5:
            failures.append(f"unexpected frame rate {result['text']}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        print((log_dir / "comfyui.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        return 1
    print("queue check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
