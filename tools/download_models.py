"""Download the two models SyncReward needs into ComfyUI's models/syncreward and verify them.

    python tools/download_models.py --models-dir <ComfyUI>/models/syncreward

* Guan123/SyncReward, folder syncreward-stage2 (about 6.2 GB, model card license Apache-2.0)
    -> <models-dir>/syncreward-stage2
* facebook/pe-av-large (about 8.9 GB, model card license Apache-2.0)
    -> <models-dir>/pe-av-large
The default revisions are the ones this node pack was verified with; the weight files are checked against the SHA-256 values published
by the Hub. Needs `huggingface_hub` (installed with ComfyUI's transformers). Nothing is downloaded when you do not run this script.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

CHECKPOINT_REPO, CHECKPOINT_REVISION = "Guan123/SyncReward", "04bc7701da8d8466c7b3aa2dec22a2a5051e6f0a"
ENCODER_REPO, ENCODER_REVISION = "facebook/pe-av-large", "0d24878d4107d64bef49e53602fc34ce6f94f6d8"
SHA256 = {
    ("syncreward-stage2", "model.safetensors"): "6c09263ac41d59baf78a181f17e4116128a337d70a03637886de3734450a32a2",
    ("syncreward-stage2", "config.json"): "0bb0e1e8b5d236352e1a828a89cd7a97af8c8f9b6dc43312f58e49d0faa71651",
    ("pe-av-large", "model.safetensors"): "7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models/syncreward folder")
    parser.add_argument("--checkpoint-revision", default=CHECKPOINT_REVISION)
    parser.add_argument("--encoder-revision", default=ENCODER_REVISION)
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download

    args.models_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.models_dir / "_download"
    snapshot_download(CHECKPOINT_REPO, revision=args.checkpoint_revision, local_dir=str(scratch / "checkpoint"), allow_patterns=["syncreward-stage2/*", "LICENSE"])
    target = args.models_dir / "syncreward-stage2"
    if not target.exists():
        (scratch / "checkpoint" / "syncreward-stage2").replace(target)
    snapshot_download(ENCODER_REPO, revision=args.encoder_revision, local_dir=str(args.models_dir / "pe-av-large"))
    failed = False
    for (folder, name), expected in SHA256.items():
        path = args.models_dir / folder / name
        if not path.is_file():
            print(f"MISSING  {folder}/{name}")
            failed = True
            continue
        if (folder == "pe-av-large" and args.encoder_revision != ENCODER_REVISION) or (folder != "pe-av-large" and args.checkpoint_revision != CHECKPOINT_REVISION):
            print(f"skipped  {folder}/{name} (other revision)")
            continue
        ok = sha256(path) == expected
        failed |= not ok
        print(f"{'ok      ' if ok else 'MISMATCH'} {folder}/{name}")
    print(f"models folder: {args.models_dir} (you may delete the _download sub-folder)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
