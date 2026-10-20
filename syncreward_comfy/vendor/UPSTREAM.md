# Vendored upstream code

`syncreward/` is a copy of the `syncreward` package of https://github.com/WeChatCV/SyncReward

- commit: `f29fbdc2bb2dd3d59644ffd3c60d207dd3e62615` (`main`, 2026-10-14)
- license: Apache License 2.0 (`LICENSE` next to this file is the upstream file)
- `MANIFEST.sha256` lists the SHA-256 of every vendored file; `tests/test_vendor.py` checks it, and, when a checkout is
  available (CI), compares the files byte for byte with the pinned upstream commit (the changed file after applying
  `scripts/vendor_patch.py`).

## The two changes

Both changed files start with a one-line change notice. All other functions of these files and all other files are unchanged; the
test compares the syntax trees and allows a difference only in the two functions below.

- `syncreward/utils.py` (since 0.1.2): the body of `setup_distributed()` is replaced by `raise NotImplementedError(...)`.
  `setup_distributed()` is the torchrun entry point of upstream training; it reads the `WORLD_SIZE` and `LOCAL_RANK` environment
  variables, which the Comfy Registry's automated scan flags in a node package. The node pack never calls it (`model.py` imports only
  `all_gather_list` from this file).
- `syncreward/model.py` (since 0.1.3): `load_state_dict()` reads a training checkpoint (`.pt`) with `weights_only=True` instead of
  `weights_only=False`, so loading cannot run pickled code. The Comfy Registry review flagged the upstream call as insecure
  deserialization. The nodes load only `model.safetensors` (they list only checkpoint folders that contain it), so their results do
  not change; a training checkpoint that stores other Python objects is refused.

All behaviour changes live outside this folder (`../runtime.py`, `../media.py`).
