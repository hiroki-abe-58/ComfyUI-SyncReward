"""Compare this pack's decoding and scoring with the official SyncReward code, clip by clip, in ONE Python environment.

    python scripts/parity_vs_official.py --upstream <SyncReward checkout> --checkpoint <syncreward-stage2 folder> \
        --encoder <pe-av-large folder> [--ffmpeg-dll-dir <FFmpeg shared bin>] a.mp4 b.mp4 ...

Needs the official requirements (torchaudio and torchcodec, which on Windows needs FFmpeg's shared DLLs: pass their folder with
--ffmpeg-dll-dir) plus PyAV. For every clip it decodes the file with the official `load_av` and with this pack's `media.decode_file`,
compares the arrays (frames, frame rate, mono audio at 48 kHz), then scores both with their own model instance (official
`build_reward_model` + `prepare_inputs` as in `score.py`; this pack's runtime) and compares the rewards bit for bit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--ffmpeg-dll-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.ffmpeg_dll_dir and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(args.ffmpeg_dll_dir))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.upstream))
    from syncreward.data import load_av
    from syncreward.model import build_reward_model, load_processor, prepare_inputs, processor_sample_rate

    from syncreward_comfy import media, runtime

    device = torch.device(args.device)
    processor = load_processor(str(args.encoder))
    sample_rate = processor_sample_rate(processor)
    official = build_reward_model(str(args.encoder), checkpoint=str(args.checkpoint)).to(device).eval()
    node = runtime.acquire(runtime.make_handle(str(args.checkpoint), str(args.encoder), args.device, True))
    print(json.dumps({"torch": torch.__version__, "node_timm_labels_corrected": node.info["timm_labels_corrected"]}))

    ns = SimpleNamespace(num_segments=18, segment_duration=0.5, segment_stride=0.25)
    rows = []
    for path in args.videos:
        frames_o, fps_o, audio_o = load_av(str(path), sample_rate)
        frames_n, fps_n, own = media.decode_file(str(path))
        audio_n = media.mono_audio(own, sample_rate)
        inputs = prepare_inputs(processor, {"video": [frames_o], "audio": [audio_o], "fps": [fps_o]}, ns, "center", device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            reward_o = float(official(inputs, encode_batch_size=18).float()[0])
        reward_n = runtime.score(node, frames_n, fps_n, audio_n)["reward"]
        row = {
            "video": path.name,
            "frames_identical": bool(frames_o.shape == frames_n.shape and np.array_equal(frames_o, frames_n)),
            "fps_identical": fps_o == fps_n,
            "audio_samples": [int(audio_o.shape[0]), int(audio_n.shape[0])],
            "audio_identical": bool(audio_o.shape == audio_n.shape and np.array_equal(audio_o, audio_n)),
            "official_reward": reward_o,
            "node_reward": reward_n,
            "identical": reward_o == reward_n,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {"clips": len(rows), "frames_identical": sum(r["frames_identical"] for r in rows), "audio_identical": sum(r["audio_identical"] for r in rows),
               "reward_identical": sum(r["identical"] for r in rows)}
    print(json.dumps(summary))
    return 0 if summary["reward_identical"] == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
