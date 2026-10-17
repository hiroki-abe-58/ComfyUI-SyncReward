"""Loading and running SyncReward inside ComfyUI.

The model code is the official repository (WeChatCV/SyncReward, commit f29fbdc), vendored unmodified in `vendor/syncreward`:
`build_reward_model` / `load_processor` build the model exactly as `score.py` does, `prepare_inputs` cuts the 18 overlapping
0.5 s segments and runs the PE-AV processor, and the reward model is called under the same bfloat16 autocast. What this module
adds is only the glue: a resident model slot, and decoded ComfyUI media (see media.py) in place of `load_av`'s file decoding.
"""

from __future__ import annotations

import gc
import json
import os
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from . import media

UPSTREAM_COMMIT = "f29fbdc2bb2dd3d59644ffd3c60d207dd3e62615"
APPROX_VRAM_BYTES = 8 * 1024**3

_LOCK = threading.RLock()
_SLOT: "Loaded | None" = None


@dataclass
class Loaded:
    key: tuple
    model: torch.nn.Module
    processor: object
    sample_rate: int
    device: torch.device
    segments: dict
    info: dict = field(default_factory=dict)


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if choice == "cuda":
            raise RuntimeError("device=cuda was requested but no CUDA device is available")
        return torch.device("cpu")
    raise ValueError(f"unknown device {choice!r}")


def make_handle(checkpoint: str, encoder: str, device: str, keep_loaded: bool) -> dict:
    return {"checkpoint": checkpoint, "encoder": encoder, "device": str(resolve_device(device)), "keep_loaded": bool(keep_loaded)}


def release() -> None:
    global _SLOT
    with _LOCK:
        _SLOT = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _free_vram(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        import comfy.model_management as mm

        mm.free_memory(APPROX_VRAM_BYTES, device)
    except Exception:  # best effort only
        pass


def _timm_class_counts(node):
    """`num_classes` of every timm_wrapper entry of a raw config.json (the vision tower's projection width)."""
    if isinstance(node, dict):
        if node.get("model_type") == "timm_wrapper" and "num_classes" in node:
            yield int(node["num_classes"])
        for value in node.values():
            yield from _timm_class_counts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _timm_class_counts(value)


def _fix_timm_labels(config, encoder_path: str) -> int:
    """Give the timm vision sub-config back the `num_classes` its config.json states.

    transformers 5.20 reads this nested timm_wrapper config with the default 2 labels instead of 1024 (5.1.0, the version the official
    repository pins, reads 1024), which makes the vision head and projection the wrong shape. Returns how many entries were corrected.
    """
    with open(os.path.join(encoder_path, "config.json"), encoding="utf-8") as handle:
        wanted = list(_timm_class_counts(json.load(handle)))
    fixed = 0

    def walk(cfg):
        nonlocal fixed
        for value in vars(cfg).values():
            if hasattr(value, "model_type") and hasattr(value, "to_dict"):
                if value.model_type == "timm_wrapper" and wanted and value.num_labels != wanted[0]:
                    value.num_labels = wanted[0]
                    fixed += 1
                walk(value)

    walk(config)
    return fixed


def _build_model(encoder_path: str, checkpoint_path: str):
    """`syncreward.model.build_reward_model` (official), except that the PE-AV config is checked first (see _fix_timm_labels)."""
    from transformers import AutoConfig, PeAudioVideoModel

    from .vendor.syncreward.model import ARCH_KEYS, PeAVEncoder, SyncRewardModel, load_config, load_state_dict, remap_legacy_keys

    config = AutoConfig.from_pretrained(encoder_path)
    fixed = _fix_timm_labels(config, encoder_path)
    full = PeAudioVideoModel.from_pretrained(encoder_path, config=config)
    encoder = PeAVEncoder(full.audio_model.audio_encoder, full.video_model.video_encoder, full.audio_model.audio_head, full.video_model.video_head)
    del full
    arch = {k: v for k, v in load_config(checkpoint_path).items() if k in ARCH_KEYS}
    model = SyncRewardModel(encoder, **arch)
    model.load_state_dict(remap_legacy_keys(load_state_dict(checkpoint_path)), strict=True)
    return model, fixed


def acquire(handle: dict) -> Loaded:
    global _SLOT
    key = (handle["checkpoint"], handle["encoder"], handle["device"])
    with _LOCK:
        if _SLOT is not None and _SLOT.key == key:
            return _SLOT
        release()
        from .vendor.syncreward.model import load_config, load_processor, processor_sample_rate

        device = torch.device(handle["device"])
        _free_vram(device)
        started = time.perf_counter()
        processor = load_processor(handle["encoder"])
        model, timm_fixed = _build_model(handle["encoder"], handle["checkpoint"])
        model.to(device).eval()
        config = load_config(handle["checkpoint"])
        segments = {
            "num_segments": int(model.num_segments),
            "segment_duration": float(config.get("segment_duration", 0.5)),
            "segment_stride": float(config.get("segment_stride", 0.25)),
            "crop": str(config.get("crop", "center")),
        }
        info = {
            "checkpoint": os.path.basename(handle["checkpoint"]),
            "encoder": os.path.basename(handle["encoder"]),
            "device": handle["device"],
            "load_seconds": round(time.perf_counter() - started, 2),
            "parameters_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "timm_labels_corrected": timm_fixed,
            **segments,
        }
        _SLOT = Loaded(key=key, model=model, processor=processor, sample_rate=processor_sample_rate(processor), device=device, segments=segments, info=info)
        return _SLOT


def score(loaded: Loaded, frames, fps: float, audio, encode_batch_size: int = 18) -> dict:
    """Reward of one clip. `frames` uint8 [T,H,W,3], `audio` mono float32 at `loaded.sample_rate` (see media.video_arrays)."""
    from .vendor.syncreward.data import MIN_AUDIO_SAMPLES, MIN_VIDEO_FRAMES
    from .vendor.syncreward.model import prepare_inputs

    if frames.shape[0] < MIN_VIDEO_FRAMES or audio.shape[0] < MIN_AUDIO_SAMPLES:
        raise media.MediaError("the clip is too short to score (needs at least 2 frames and 0.1 s of audio)")
    seg = loaded.segments
    span = seg["segment_duration"] + (seg["num_segments"] - 1) * seg["segment_stride"]
    args = SimpleNamespace(num_segments=seg["num_segments"], segment_duration=seg["segment_duration"], segment_stride=seg["segment_stride"])
    device = loaded.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    # the same sequence as score.py: processor inputs, then the model under bfloat16 autocast on CUDA
    inputs = prepare_inputs(loaded.processor, {"video": [frames], "audio": [audio], "fps": [fps]}, args, seg["crop"], device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        reward = float(loaded.model(inputs, encode_batch_size=int(encode_batch_size)).float()[0])
    elapsed = time.perf_counter() - started
    video_seconds = frames.shape[0] / fps
    audio_seconds = audio.shape[0] / loaded.sample_rate
    report = {
        "reward": reward,
        "video_seconds": round(video_seconds, 3),
        "audio_seconds": round(audio_seconds, 3),
        "fps": round(fps, 4),
        "frames": int(frames.shape[0]),
        "window_seconds": round(span, 3),
        "shorter_than_window": bool(min(video_seconds, audio_seconds) < span),
        "scoring_seconds": round(elapsed, 3),
    }
    if device.type == "cuda":
        report["peak_vram_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
    return report


def report_json(loaded_info: dict, report: dict) -> str:
    return json.dumps({"model": {k: loaded_info[k] for k in ("checkpoint", "encoder", "device")}, **report}, indent=1)
