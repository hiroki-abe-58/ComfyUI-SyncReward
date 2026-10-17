"""Score video files through a RUNNING ComfyUI queue with these nodes, optionally next to the official `score.py` results.

    python scripts/queue_scores.py --server http://127.0.0.1:8188 --comfy-input <ComfyUI/input> --out scores.json a.mp4 b.mp4 \
        [--official official.jsonl]

Every video is copied into the ComfyUI input folder as `syncreward_<name>`, scored with the shipped `workflows/api/score_basic.json`
(Load Video -> SyncReward Score) through `POST /prompt`, and the reward is printed. `--official` takes the jsonl written by the
official `python score.py --output official.jsonl ...` (lines of {"video_path", "reward"}; matched by file name) and adds the
difference per clip and a summary.
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

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-input", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--official", type=Path)
    parser.add_argument("--checkpoint", default="syncreward-stage2")
    parser.add_argument("--encoder", default="pe-av-large")
    args = parser.parse_args(argv)

    official = {}
    if args.official:
        for line in args.official.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                official[Path(row["video_path"]).name] = float(row["reward"])

    workflow_template = json.loads((ROOT / "workflows" / "api" / "score_basic.json").read_text(encoding="utf-8"))
    rows, failed = [], 0
    for video in args.videos:
        name = f"syncreward_{video.name}"
        shutil.copyfile(video, args.comfy_input / name)
        workflow = json.loads(json.dumps(workflow_template))
        workflow["1"]["inputs"].update({"checkpoint": args.checkpoint, "encoder": args.encoder})
        workflow["2"]["inputs"]["file"] = name
        result = queue_client.run(args.server, workflow, None, 900)
        row = {"video": video.name, "status": result["status"], "error": result.get("error")}
        if result["status"] == "success":
            report = json.loads(next(v for v in result["text"].values() if "{" in v[0])[0])
            row.update({"reward": report["reward"], "scoring_seconds": report["scoring_seconds"], "peak_vram_allocated_gib": report.get("peak_vram_allocated_gib")})
            if video.name in official:
                row["official"] = official[video.name]
                row["difference"] = round(report["reward"] - official[video.name], 6)
        else:
            failed += 1
        rows.append(row)
        print(json.dumps(row), flush=True)
    diffs = [abs(r["difference"]) for r in rows if "difference" in r]
    summary = {"videos": len(rows), "failed": failed}
    if diffs:
        summary.update({"compared": len(diffs), "max_abs_difference": max(diffs), "mean_abs_difference": round(statistics.mean(diffs), 6), "identical": sum(d == 0 for d in diffs)})
    print(json.dumps(summary))
    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
