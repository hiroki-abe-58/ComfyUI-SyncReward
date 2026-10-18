"""Test setup. References come from environment variables, never from this repository:

SYNCREWARD_COMFYUI_DIR    ComfyUI checkout (v0.38.0 or later)
SYNCREWARD_UPSTREAM_DIR   WeChatCV/SyncReward checkout (the commit in syncreward_comfy/vendor/UPSTREAM.md)
SYNCREWARD_PEAV_CONFIG    the config.json of facebook/pe-av-large (revision in the CI file; about 50 kB, no weights)
SYNCREWARD_REQUIRE_REFERENCES=1  fail instead of skip when one of them is missing (CI sets this)
No model weights are used: the model tests run a stub encoder on the CPU.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
REQUIRE = os.environ.get("SYNCREWARD_REQUIRE_REFERENCES") == "1"


def _path(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _path("SYNCREWARD_COMFYUI_DIR")
UPSTREAM = _path("SYNCREWARD_UPSTREAM_DIR")
PEAV_CONFIG = _path("SYNCREWARD_PEAV_CONFIG")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (SYNCREWARD_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (SYNCREWARD_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def upstream_dir():
    return _need(UPSTREAM, "SyncReward checkout (SYNCREWARD_UPSTREAM_DIR)")


@pytest.fixture(scope="session")
def peav_config():
    return Path(_need(PEAV_CONFIG, "pe-av-large config.json (SYNCREWARD_PEAV_CONFIG)"))


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory):
    from sample_video import make_sample_video

    return make_sample_video(tmp_path_factory.mktemp("media") / "example.mp4")
