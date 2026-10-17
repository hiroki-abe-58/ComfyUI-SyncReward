"""Does the reward fall when a video's audio is moved out of sync?  Runs through a RUNNING ComfyUI queue.

    python scripts/experiment_audio_offset.py --server http://127.0.0.1:8188 --comfy-input <ComfyUI/input> --out offsets.json \
        --offsets -1000 -500 -250 -100 0 100 250 500 1000  a.mp4 b.mp4 ...

For every clip and offset a small workflow (Load Video -> SyncReward Shift Audio -> SyncReward Score) is queued with POST /prompt;
a positive offset delays the sound, a negative one advances it, the picture and the clip length stay the same. Prints the
rewards per clip, the mean per offset and, for each clip, whether the unshifted reward is at least as high as every shifted one.
The clips are only read; nothing is uploaded or stored besides the numbers written to --out.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import queue_client  # noqa: E402


def workflow(name: str, offset: float, checkpoint: str, encoder: str) -> dict:
    return {
        "1": {"class_type": "SyncRewardLoader", "inputs": {"checkpoint": checkpoint, "encoder": encoder, "device": "auto", "keep_loaded": True}},
        "2": {"class_type": "LoadVideo", "inputs": {"file": name}},
        "3": {"class_type": "SyncRewardShiftAudio", "inputs": {"video": ["2", 0], "offset_ms": float(offset)}},
        "4": {"class_type": "SyncRewardScore", "inputs": {"model": ["1", 0], "video": ["3", 0], "encode_batch_size": 18}},
        "5": {"class_type": "PreviewAny", "inputs": {"source": ["4", 0]}},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-input", type=Path, required=True)
    parser.add_argument("--offsets", nargs="+", type=float, default=[-1000, -500, -250, -100, 0, 100, 250, 500, 1000])
    parser.add_argument("--checkpoint", default="syncreward-stage2")
    parser.add_argument("--encoder", default="pe-av-large")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    table: dict[str, dict[str, float]] = {}
    for video in args.videos:
        name = f"syncreward_{video.name}"
        shutil.copyfile(video, args.comfy_input / name)
        table[video.name] = {}
        for offset in args.offsets:
            result = queue_client.run(args.server, workflow(name, offset, args.checkpoint, args.encoder), None, 900)
            if result["status"] != "success":
                raise SystemExit(f"{video.name} @ {offset} ms failed: {result.get('error') or result['status']}")
            table[video.name][str(int(offset))] = float(next(iter(result["text"].values()))[0])
        print(json.dumps({video.name: table[video.name]}), flush=True)

    keys = [str(int(o)) for o in args.offsets]
    means = {k: round(statistics.mean(table[v][k] for v in table), 4) for k in keys}
    zero = "0" if "0" in keys else None
    summary = {"offsets_ms": [int(o) for o in args.offsets], "mean_reward": means}
    if zero:
        summary["clips_where_unshifted_is_highest"] = sum(all(table[v][zero] >= table[v][k] for k in keys) for v in table)
        summary["clips"] = len(table)
        summary["share_of_shifted_runs_below_unshifted"] = round(
            sum(table[v][k] < table[v][zero] for v in table for k in keys if k != zero) / (len(table) * (len(keys) - 1)), 3
        )
    print(json.dumps(summary))
    if args.out:
        args.out.write_text(json.dumps({"rewards": table, "summary": summary}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
