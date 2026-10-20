"""The node pack's two documented changes to the vendored upstream code (maintainer tool and test helper).

    python scripts/vendor_patch.py --upstream <WeChatCV/SyncReward checkout at the pinned commit>

- `syncreward/utils.py`: `setup_distributed()` (the torchrun entry point of training) reads the WORLD_SIZE and LOCAL_RANK environment
  variables; the node pack never calls it, and environment reads in a node package are flagged by the Comfy Registry scan. Its body
  is replaced by an error.
- `syncreward/model.py`: `load_state_dict()` reads a training checkpoint (`.pt`) with `torch.load(weights_only=False)`, which can run
  pickled code; the nodes load only `model.safetensors`, and the Comfy Registry flagged the call (insecure deserialization). It now
  uses `weights_only=True`.

Each changed file starts with a change notice (Apache-2.0, section 4(b)). Every other vendored file is byte-identical to upstream.
tests/test_vendor.py applies `patch()` to the upstream checkout in CI and compares the bytes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "syncreward_comfy" / "vendor"
PATCHED = {"syncreward/utils.py", "syncreward/model.py"}
NOTICE = "# This file was changed by ComfyUI-SyncReward; syncreward_comfy/vendor/UPSTREAM.md describes the change.\n"
_OLD = '''def setup_distributed(backend=None):
    """Initialise torch.distributed when launched by torchrun with WORLD_SIZE > 1."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend or ("nccl" if torch.cuda.is_available() else "gloo"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return get_rank(), get_world_size(), device
'''
_NEW = '''def setup_distributed(backend=None):
    """Initialise torch.distributed when launched by torchrun (training; not part of the ComfyUI node pack)."""
    raise NotImplementedError("setup_distributed is the training entry point and is not included in ComfyUI-SyncReward")
'''
_CHANGES = {
    "syncreward/utils.py": ("setup_distributed()", _OLD, _NEW),
    "syncreward/model.py": ("load_state_dict()", 'torch.load(path, map_location="cpu", weights_only=False)',
                            'torch.load(path, map_location="cpu", weights_only=True)'),
}


def patch(rel: str, text: str) -> str:
    name, old, new = _CHANGES[rel]
    if text.count(old) != 1:
        raise ValueError(f"{rel}: {name} differs from the pinned upstream version")
    return NOTICE + text.replace(old, new)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True)
    args = ap.parse_args(argv)
    for rel in sorted(PATCHED):
        data = patch(rel, (args.upstream / rel).read_bytes().decode("utf-8")).encode("utf-8")
        (VENDOR / rel).write_bytes(data)
        print("wrote", rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
