# Vendored upstream code

`syncreward/` is a copy of the `syncreward` package of https://github.com/WeChatCV/SyncReward

- commit: `f29fbdc2bb2dd3d59644ffd3c60d207dd3e62615` (`main`, 2026-10-14)
- license: Apache License 2.0 (`LICENSE` next to this file is the upstream file)
- `MANIFEST.sha256` lists the SHA-256 of every vendored file; `tests/test_vendor.py` checks it, and, when a checkout is
  available (CI), compares the files byte for byte with the pinned upstream commit (the changed file after applying
  `scripts/vendor_patch.py`).

## The one change (since 0.1.2)

`syncreward/utils.py`: the body of `setup_distributed()` is replaced by `raise NotImplementedError(...)`, and the file starts with a
one-line change notice. `setup_distributed()` is the torchrun entry point of upstream training; it reads the `WORLD_SIZE` and
`LOCAL_RANK` environment variables, which the Comfy Registry's automated scan flags in a node package. The node pack never calls it
(`model.py` imports only `all_gather_list` from this file). All other functions of `utils.py` and all other files are unchanged; the
test compares the syntax trees and allows a difference only in `setup_distributed`.

All behaviour changes live outside this folder (`../runtime.py`, `../media.py`).
