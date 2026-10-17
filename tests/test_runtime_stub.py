"""The scoring path (official segment cutting + official reward model) with a stub encoder on the CPU, and the PE-AV config fix."""

import json
import shutil

import numpy as np
import pytest

from stubs import stub_loaded
from syncreward_comfy import media, runtime


def _clip(seconds=6.0, fps=24.0, size=8):
    frames = np.random.RandomState(0).randint(0, 255, (int(seconds * fps), size, size, 3), dtype=np.uint8)
    audio = np.random.RandomState(1).randn(int(seconds * 48000)).astype(np.float32) * 0.1
    return frames, fps, audio


def test_score_is_in_range_deterministic_and_reports_the_window():
    loaded = stub_loaded()
    frames, fps, audio = _clip()
    first = runtime.score(loaded, frames, fps, audio)
    again = runtime.score(loaded, frames, fps, audio)
    assert 0.0 <= first["reward"] <= 2.0 and first["reward"] == again["reward"]
    assert first["window_seconds"] == 4.75 and first["shorter_than_window"] is False and first["frames"] == 144
    assert json.loads(runtime.report_json(loaded.info, first))["reward"] == first["reward"]


def test_audio_changes_the_score_and_short_clips_are_flagged():
    loaded = stub_loaded()
    frames, fps, audio = _clip()
    shifted = np.concatenate([np.zeros(12000, dtype=np.float32), audio[:-12000]])
    assert runtime.score(loaded, frames, fps, audio)["reward"] != runtime.score(loaded, frames, fps, shifted)["reward"]
    short = runtime.score(loaded, frames[:48], fps, audio[:96000])
    assert short["shorter_than_window"] is True


def test_clips_that_are_too_short_are_rejected():
    loaded = stub_loaded()
    frames, fps, audio = _clip()
    with pytest.raises(media.MediaError, match="too short"):
        runtime.score(loaded, frames[:1], fps, audio)
    with pytest.raises(media.MediaError, match="too short"):
        runtime.score(loaded, frames, fps, audio[:100])


def test_pe_av_config_gets_its_vision_head_width_back(peav_config, tmp_path):
    """transformers 5.20 reads the nested timm config with 2 labels; the pack restores the config.json value (1024)."""
    from transformers import AutoConfig

    shutil.copyfile(peav_config, tmp_path / "config.json")
    config = AutoConfig.from_pretrained(str(tmp_path))
    fixed = runtime._fix_timm_labels(config, str(tmp_path))
    assert fixed in (0, 1)  # 0 on transformers versions that parse it correctly
    assert list(runtime._timm_class_counts(json.loads(peav_config.read_text(encoding="utf-8")))) == [1024]

    def timm_configs(cfg):
        for value in vars(cfg).values():
            if hasattr(value, "model_type") and hasattr(value, "to_dict"):
                if value.model_type == "timm_wrapper":
                    yield value
                yield from timm_configs(value)

    found = list(timm_configs(config))
    assert found and all(c.num_labels == 1024 for c in found)


def test_release_and_resident_slot():
    runtime._SLOT = stub_loaded()
    runtime.release()
    assert runtime._SLOT is None
