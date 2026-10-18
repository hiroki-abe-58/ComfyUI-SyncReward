"""The vendored upstream package must stay byte-identical to the pinned upstream commit, apart from the one documented change of
scripts/vendor_patch.py, and the installed package must not contain environment-variable, network or process code."""

import hashlib
import re
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
    """CI checks out the pinned commit; compare the files (LF line endings on every platform). The one patched file must equal the
    upstream file with scripts/vendor_patch.py applied."""
    import vendor_patch

    upstream = Path(upstream_dir)
    for path in (VENDOR / "syncreward").rglob("*.py"):
        rel = path.relative_to(VENDOR).as_posix()
        original = (upstream / rel).read_bytes()
        expected = vendor_patch.patch(rel, original.decode("utf-8")).encode("utf-8") if rel in vendor_patch.PATCHED else original
        assert path.read_bytes() == expected, rel
    assert (VENDOR / "LICENSE").read_bytes() == (upstream / "LICENSE").read_bytes()
    assert len(list((upstream / "syncreward").rglob("*.py"))) == len(list((VENDOR / "syncreward").rglob("*.py")))


def test_the_patch_changes_only_the_training_entry_point(upstream_dir):
    import ast

    import vendor_patch

    for rel in vendor_patch.PATCHED:
        before = ast.parse((Path(upstream_dir) / rel).read_text(encoding="utf-8"))
        after = ast.parse((VENDOR / rel).read_text(encoding="utf-8"))
        changed = [a.name for a, b in zip(before.body, after.body) if ast.dump(a) != ast.dump(b)]
        assert len(before.body) == len(after.body) and changed == ["setup_distributed"]


PACKAGE_FILES = [p for p in [ROOT / "__init__.py", *(ROOT / "syncreward_comfy").rglob("*.py"), *(ROOT / "tools").rglob("*.py")]
                 if "__pycache__" not in p.parts]
FORBIDDEN = re.compile(r"os\.environ|getenv|putenv|urllib|requests\.|http\.client|socket\.|subprocess|os\.system|importlib|(?<![.\w])exec\(|(?<![.\w])eval\(|__import__")


def test_package_has_no_environment_network_or_process_code():
    """The categories of the Comfy Registry's automated scan (scripts/ and tests/ are not part of the Registry package)."""
    hits = [f"{p.relative_to(ROOT).as_posix()}:{n}: {line.strip()}" for p in PACKAGE_FILES
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if FORBIDDEN.search(line)]
    assert not hits, "\n".join(hits)
