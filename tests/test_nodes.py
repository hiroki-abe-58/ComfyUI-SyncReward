"""The node pack as ComfyUI loads it: registration, ComfyUI's own validation of the shipped workflows, and node execution with a stub
encoder on the CPU (the real weights are used in the real-queue checks documented in docs/VERIFICATION.md)."""

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

from stubs import stub_loaded

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = {"SyncRewardLoader", "SyncRewardScore", "SyncRewardSelectBest", "SyncRewardShiftAudio", "SyncRewardUnload"}


@pytest.fixture(scope="module")
def comfy_nodes(comfy_dir, sample_video, tmp_path_factory):
    import folder_paths
    import nodes

    base = tmp_path_factory.mktemp("comfy")
    input_dir = base / "input"
    input_dir.mkdir()
    shutil.copyfile(sample_video, input_dir / "example.mp4")  # the file name the shipped workflows use
    folder_paths.set_input_directory(str(input_dir))

    models = base / "models" / "syncreward"
    (models / "syncreward-stage2").mkdir(parents=True)
    (models / "syncreward-stage2" / "config.json").write_text(json.dumps({"model_type": "syncreward", "stage": 2}), encoding="utf-8")
    (models / "syncreward-stage2" / "model.safetensors").write_bytes(b"")
    (models / "pe-av-large").mkdir()
    (models / "pe-av-large" / "config.json").write_text(json.dumps({"model_type": "pe_audio_video"}), encoding="utf-8")
    for name in ("model.safetensors", "preprocessor_config.json", "processor_config.json"):
        (models / "pe-av-large" / name).write_bytes(b"")
    (models / "incomplete").mkdir()
    (models / "incomplete" / "config.json").write_text(json.dumps({"model_type": "syncreward"}), encoding="utf-8")  # no weights
    (models / "other").mkdir()
    (models / "other" / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (models / "other" / "model.safetensors").write_bytes(b"")
    folder_paths.add_model_folder_path("syncreward", str(models), is_default=True)

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    folder_paths.filename_list_cache.clear()
    return nodes


def test_registered(comfy_nodes):
    assert NODE_IDS <= set(comfy_nodes.NODE_CLASS_MAPPINGS)


def test_model_folders_are_told_apart_by_model_type_and_files(comfy_nodes):
    from syncreward_comfy import paths

    assert paths.checkpoints() == ["syncreward-stage2"]  # not "incomplete" (no weights) and not "other" (wrong model_type)
    assert paths.encoders() == ["pe-av-large"]
    for bad in ("..", "../x", "incomplete", "missing", "other"):
        with pytest.raises(FileNotFoundError):
            paths.checkpoint_path(bad)
    with pytest.raises(FileNotFoundError):
        paths.encoder_path("syncreward-stage2")


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "workflows" / "api").glob("*.json")))
def test_api_workflows_pass_comfy_validation(comfy_nodes, name):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / name).read_text(encoding="utf-8"))
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert valid, (error, node_errors)


def test_validation_rejects_out_of_range_values(comfy_nodes):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / "score_basic.json").read_text(encoding="utf-8"))
    prompt["3"]["inputs"]["encode_batch_size"] = 0
    valid, _, _, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert not valid and "3" in node_errors


def node_runtime(comfy_nodes):
    """The runtime module the NODES use. ComfyUI imports the pack under a path-based name, so it is a different module object from
    `syncreward_comfy.runtime` as the tests import it; patches must go to this one."""
    module_name = comfy_nodes.NODE_CLASS_MAPPINGS["SyncRewardScore"].__module__.rsplit(".", 1)[0] + ".runtime"
    return sys.modules[module_name]


@pytest.fixture()
def stub_model(monkeypatch, comfy_nodes):
    runtime = node_runtime(comfy_nodes)
    loaded = stub_loaded()
    monkeypatch.setattr(runtime, "acquire", lambda handle: loaded)
    return {"checkpoint": "stub", "encoder": "stub", "device": "cpu", "keep_loaded": True}


def test_score_node_returns_a_reward_and_a_report(comfy_nodes, stub_model, sample_video):
    from comfy_api.latest import VideoFromFile

    out = comfy_nodes.NODE_CLASS_MAPPINGS["SyncRewardScore"].execute(stub_model, VideoFromFile(str(sample_video)), 18)
    reward, report = out.args
    assert 0.0 <= reward <= 2.0
    parsed = json.loads(report)
    assert parsed["reward"] == reward and parsed["frames"] == 30 and parsed["shorter_than_window"] is True


def test_audio_input_replaces_the_video_track(comfy_nodes, stub_model, sample_video):
    import torch
    from comfy_api.latest import VideoFromFile

    node = comfy_nodes.NODE_CLASS_MAPPINGS["SyncRewardScore"]
    video = VideoFromFile(str(sample_video))
    own = node.execute(stub_model, video, 18).args[0]
    silent = node.execute(stub_model, video, 18, audio={"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000}).args[0]
    assert own != silent


def test_shift_audio_node_builds_a_video_that_scores_differently(comfy_nodes, stub_model, sample_video):
    from comfy_api.latest import VideoFromFile

    shift = comfy_nodes.NODE_CLASS_MAPPINGS["SyncRewardShiftAudio"]
    score = comfy_nodes.NODE_CLASS_MAPPINGS["SyncRewardScore"]
    video = VideoFromFile(str(sample_video))
    zero = shift.execute(video, 0.0).args[0]
    moved = shift.execute(video, 250.0).args[0]
    base = score.execute(stub_model, video, 18).args[0]
    assert score.execute(stub_model, zero, 18).args[0] == pytest.approx(base, abs=1e-6)  # a zero offset changes nothing
    assert score.execute(stub_model, moved, 18).args[0] != base


def test_select_best_returns_the_highest_reward_and_breaks_ties_to_the_lowest_number(comfy_nodes, stub_model, sample_video):
    from comfy_api.latest import VideoFromFile

    shift = comfy_nodes.NODE_CLASS_MAPPINGS["SyncRewardShiftAudio"]
    select = comfy_nodes.NODE_CLASS_MAPPINGS["SyncRewardSelectBest"]
    score = comfy_nodes.NODE_CLASS_MAPPINGS["SyncRewardScore"]
    original = VideoFromFile(str(sample_video))
    candidates = [shift.execute(original, ms).args[0] for ms in (400.0, 0.0, -300.0)]
    rewards = [score.execute(stub_model, c, 18).args[0] for c in candidates]
    best_video, index, best_reward, scores = select.execute(stub_model, candidates[0], candidates[1], 18, video_3=candidates[2]).args
    expected = max(range(3), key=lambda i: (rewards[i], -i))
    assert index == expected + 1 and best_reward == pytest.approx(rewards[expected], abs=1e-6) and best_video is candidates[expected]
    assert set(json.loads(scores)["rewards"]) == {"video_1", "video_2", "video_3"}
    _, tie_index, _, _ = select.execute(stub_model, candidates[1], candidates[1], 18).args
    assert tie_index == 1


def test_missing_audio_is_reported_clearly(comfy_nodes, stub_model):
    from fractions import Fraction

    import torch
    from comfy_api.latest import VideoComponents, VideoFromComponents

    silent = VideoFromComponents(VideoComponents(images=torch.zeros(30, 8, 8, 3), frame_rate=Fraction(24)))
    with pytest.raises(ValueError, match="no audio track"):
        comfy_nodes.NODE_CLASS_MAPPINGS["SyncRewardScore"].execute(stub_model, silent, 18)
    with pytest.raises(ValueError, match="no audio track"):
        comfy_nodes.NODE_CLASS_MAPPINGS["SyncRewardShiftAudio"].execute(silent, 100.0)


def test_keep_loaded_false_releases_after_the_job(comfy_nodes, monkeypatch, sample_video):
    from comfy_api.latest import VideoFromFile

    runtime = node_runtime(comfy_nodes)
    loaded = stub_loaded()
    monkeypatch.setattr(runtime, "acquire", lambda handle: loaded)
    monkeypatch.setattr(runtime, "_SLOT", loaded)
    handle = {"checkpoint": "stub", "encoder": "stub", "device": "cpu", "keep_loaded": False}
    comfy_nodes.NODE_CLASS_MAPPINGS["SyncRewardScore"].execute(handle, VideoFromFile(str(sample_video)), 18)
    assert runtime._SLOT is None
    comfy_nodes.NODE_CLASS_MAPPINGS["SyncRewardUnload"].execute()  # safe when nothing is loaded
