# Vendored upstream code

`syncreward/` is an UNMODIFIED copy of the `syncreward` package of https://github.com/WeChatCV/SyncReward

- commit: `f29fbdc2bb2dd3d59644ffd3c60d207dd3e62615` (`main`, 2026-10-14)
- license: Apache License 2.0 (`LICENSE` next to this file is the upstream file)
- `MANIFEST.sha256` lists the SHA-256 of every vendored file; `tests/test_vendor.py` checks it, and, when a checkout is
  available (CI), compares the files byte for byte with the pinned upstream commit.

All behaviour changes live outside this folder (`../runtime.py`, `../media.py`).
