"""Negative controls for the reward: what happens when the audio is clearly wrong?  (in-process; loads the real weights)

    python scripts/experiment_controls.py --comfyui <ComfyUI checkout> --checkpoint <syncreward-stage2 folder> \
        --encoder <pe-av-large folder> --out controls.json a.mp4 b.mp4 ...

Per clip it scores the original and: silence, white noise of the same loudness, the audio played backwards, the audio of another
clip (the one three places later in the list, cut or repeated to the same length) and the audio delayed by 2 s and 3 s.
Needs at least two clips. The clips are only read; the numbers are printed and written to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--comfyui", type=Path, help="ComfyUI checkout (supplies comfy.audio.resample)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if len(args.videos) < 2:
        parser.error("at least two clips are needed (the 'other clip' control)")

    sys.path.insert(0, str(ROOT))
    if args.comfyui:
        sys.path.insert(1, str(args.comfyui))
    from syncreward_comfy import media, runtime

    loaded = runtime.acquire(runtime.make_handle(str(args.checkpoint), str(args.encoder), args.device, True))
    data = {}
    for path in args.videos:
        frames, fps, own = media.decode_file(str(path))
        data[path.name] = (frames, fps, media.mono_audio(own, loaded.sample_rate))
    names = list(data)
    rng = np.random.RandomState(0)
    rows = {}
    for i, name in enumerate(names):
        frames, fps, audio = data[name]
        other = np.resize(data[names[(i + 3) % len(names)]][2], audio.shape).astype(np.float32)
        rms = float(np.sqrt((audio ** 2).mean()))
        variants = {
            "original": audio,
            "silence": np.zeros_like(audio),
            "white_noise_same_rms": (rng.randn(*audio.shape) * rms).astype(np.float32),
            "time_reversed": audio[::-1].copy(),
            "other_clip_audio": other,
        }
        for ms in (2000, 3000):
            k = min(int(ms / 1000 * loaded.sample_rate), audio.shape[0] - 1)
            variants[f"shift+{ms}ms"] = np.concatenate([np.zeros(k, dtype=np.float32), audio[:-k]])
        rows[name] = {label: round(runtime.score(loaded, frames, fps, a)["reward"], 4) for label, a in variants.items()}
        print(json.dumps({name: rows[name]}), flush=True)
    labels = list(next(iter(rows.values())))
    mean = {label: round(float(np.mean([rows[n][label] for n in names])), 4) for label in labels}
    print(json.dumps({"mean": mean}))
    if args.out:
        args.out.write_text(json.dumps({"rewards": rows, "mean": mean}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
