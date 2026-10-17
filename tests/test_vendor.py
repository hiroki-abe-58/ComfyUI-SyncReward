"""The vendored upstream package must stay byte-identical to the pinned upstream commit."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "syncreward_comfy" / "vendor"
UPSTREAM_COMMIT = "f29fbdc2bb2dd3d59644ffd3c60d207dd3e62615"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_matches_every_vendored_file():
    listed = {}
    for line in (VENDOR / "MANIFEST.sha256").read_text(encoding="ascii").splitlines():
        digest, _, name = line.partition("  ")
        listed[name] = digest
    actual = {p.relative_to(VENDOR).as_posix(): _sha(p) for p in (VENDOR / "syncreward").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    actual["LICENSE"] = _sha(VENDOR / "LICENSE")
    assert actual == listed


def test_vendor_notes_pin_the_commit():
    assert UPSTREAM_COMMIT in (VENDOR / "UPSTREAM.md").read_text(encoding="utf-8")


def test_byte_identical_to_upstream_checkout(upstream_dir):
    """CI checks out the pinned commit; compare the files (LF line endings on every platform)."""
    upstream = Path(upstream_dir)
    for path in (VENDOR / "syncreward").rglob("*.py"):
        rel = path.relative_to(VENDOR).as_posix()
        assert path.read_bytes() == (upstream / rel).read_bytes(), rel
    assert (VENDOR / "LICENSE").read_bytes() == (upstream / "LICENSE").read_bytes()
    assert len(list((upstream / "syncreward").rglob("*.py"))) == len(list((VENDOR / "syncreward").rglob("*.py")))
