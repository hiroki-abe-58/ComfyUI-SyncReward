"""Fetch single clips of SyncReward-Data from its (stored, uncompressed) zip archives with HTTP range requests.

    python scripts/fetch_bench_clips.py --data-dir <dir with bench/index jsonl> --out <dir> [--limit N]

The dataset (Guan123/SyncReward-Data, annotations Apache-2.0, clips produced by third-party generators whose own terms also
apply, research use) stores 57,099 clips in 213 GB of zip files. The 1,000 SyncReward-Bench clips are spread over 15 of them. The
zips are 'stored', so one member can be read without the rest: this script reads each archive's central directory and then
only the members it needs (about 4 GB for the whole benchmark). The clips are for local evaluation; do not republish them.
Clips are written to <out>/generated/<generator>/<name>.mp4, so `--video_root <out>` works with the official evaluate.py. Files already
present with the right size are skipped. Pinned to one dataset revision.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REVISION = "a7904b6d5901563ce4aad55d5daad4e601c58630"
BASE = f"https://huggingface.co/datasets/Guan123/SyncReward-Data/resolve/{REVISION}/"


class RangeFile(io.RawIOBase):
    """Read-only file over an HTTP URL; every read is one Range request (zipfile's reads are few and sequential)."""

    def __init__(self, url: str):
        self.base = url
        self.pos = 0
        self._resolve()

    def _resolve(self):
        """Follow the redirect once and keep the final (CDN) address, so that later reads cost one round trip."""
        request = urllib.request.Request(self.base, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            self.url = response.geturl()
            self.size = int(response.headers["Content-Range"].rsplit("/", 1)[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        want = min(len(buffer), self.size - self.pos)
        if want <= 0:
            return 0
        got = 0
        while got < want:
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos + got}-{self.pos + want - 1}"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    chunk = response.read()
            except urllib.error.HTTPError as err:
                if err.code not in (400, 401, 403):  # the signed CDN address expired
                    raise
                self._resolve()
                continue
            if not chunk:
                raise IOError("empty range response")
            buffer[got:got + len(chunk)] = chunk
            got += len(chunk)
        self.pos += got
        return got


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="folder with bench_syncreward_bench.jsonl and videos_index.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids", type=Path, help="optional text file with bench ids to fetch (default: all)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=5, help="archives read in parallel")
    args = parser.parse_args(argv)

    bench = [json.loads(line) for line in (args.data_dir / "bench_syncreward_bench.jsonl").read_text(encoding="utf-8").splitlines()]
    index = {}
    for line in (args.data_dir / "videos_index.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        index[row["video_path"]] = row["zip"]
    if args.ids:
        wanted = set(args.ids.read_text(encoding="utf-8").split())
        bench = [row for row in bench if row["id"] in wanted]
    if args.limit:
        bench = bench[:args.limit]
    by_zip = collections.defaultdict(list)
    for row in bench:
        by_zip[index[row["video_path"]]].append(row)

    args.out.mkdir(parents=True, exist_ok=True)

    def fetch_archive(item):
        name, rows = item
        started = time.time()
        archive = zipfile.ZipFile(io.BufferedReader(RangeFile(BASE + name), buffer_size=8 << 20))
        for row in rows:
            target = args.out / row["video_path"]  # generated/<model>/<name>.mp4: the layout --video_root expects
            target.parent.mkdir(parents=True, exist_ok=True)
            info = archive.getinfo(row["video_path"])
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            with archive.open(info) as source, open(target.with_suffix(".part"), "wb") as sink:
                while True:
                    block = source.read(1 << 20)
                    if not block:
                        break
                    sink.write(block)
            target.with_suffix(".part").replace(target)
        print(f"{name}: {len(rows)} clips in {time.time() - started:.0f} s", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch_archive, sorted(by_zip.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
