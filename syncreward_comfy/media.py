"""ComfyUI VIDEO / AUDIO values -> the arrays the official SyncReward code works on, and a helper that shifts a video's audio.

The official `syncreward.data.load_av` decodes a file into (frames uint8 [T, H, W, C], fps, mono float32 audio at the
encoder's sample rate). Here the same triple is built from a ComfyUI VIDEO, so the model code downstream (`prepare_inputs`, the
reward model) is the official one, unchanged.

An unmodified `Load Video` VIDEO is decoded from its file exactly like `load_av` does it (see `decode_file`). ComfyUI's own
decoded components are not used for it because, for videos without colour tags, ComfyUI converts YUV to RGB with a different matrix
than torchcodec (pixel values differ by up to about 40 of 255 and the reward by a few hundredths). Any other VIDEO (trimmed, cropped,
built from images or by another node) is converted from its components.
"""

from __future__ import annotations

import struct
from fractions import Fraction

import numpy as np
import torch


class MediaError(ValueError):
    """The video cannot be scored (no audio, too short, ...)."""


def _resample(waveform: torch.Tensor, orig: int, new: int) -> torch.Tensor:
    try:
        from comfy.audio import resample  # ComfyUI's port of torchaudio.functional.resample (same defaults)
    except ImportError:
        try:
            from torchaudio.functional import resample
        except ImportError as exc:
            raise RuntimeError("Resampling needs a ComfyUI with comfy.audio.resample or torchaudio") from exc
    return resample(waveform, orig, new)


def mono_audio(audio: dict, sample_rate: int) -> np.ndarray:
    """AUDIO dict ({"waveform": [B, C, S], "sample_rate"}) -> mono float32 numpy at `sample_rate` (first batch item)."""
    waveform = audio["waveform"]
    if waveform.dim() != 3 or waveform.shape[-1] < 1:
        raise MediaError("the audio is empty or malformed")
    wav = waveform[0].detach().float().cpu().mean(dim=0, keepdim=True)
    source = int(audio["sample_rate"])
    if source != sample_rate:
        wav = _resample(wav, source, sample_rate)
    return wav[0].numpy().astype(np.float32, copy=False)


def _plain_file_source(video):
    """The file behind an unmodified `Load Video` VIDEO (no trim, no crop), else None."""
    try:
        from comfy_api.latest import VideoFromFile
    except ImportError:
        return None
    if type(video) is not VideoFromFile:
        return None
    try:
        if video.get_active_trim_window() != (0.0, 0.0) or getattr(video, "_VideoFromFile__crop", None) is not None:
            return None
        return video.get_stream_source()
    except Exception:  # noqa: BLE001 - anything unexpected: use the generic path
        return None


def _leading_skip(frame) -> int:
    """Samples to drop from the start of a decoded audio frame (FFmpeg's SKIP_SAMPLES side data, start part), else 0."""
    try:
        for key, value in frame.side_data.items():
            if str(key).endswith("SKIP_SAMPLES"):
                return int(struct.unpack("<I", bytes(value)[:4])[0])
    except Exception:  # noqa: BLE001 - side data not available in this PyAV
        pass
    return 0


def decode_file(source, with_audio: bool = True):
    """Decode a video file like the official `load_av`: (frames uint8 [T,H,W,3], fps, AUDIO dict at the file's own rate | None).

    Frames are the decoder's plain 8-bit RGB conversion (what torchcodec returns). The frame rate is frames / duration of the video
    stream (torchcodec's `average_fps`). The audio is every decoded sample (not trimmed to the video length) after the leading priming
    samples that the first packet says to skip, all channels.
    """
    import av

    if hasattr(source, "seek"):
        source.seek(0)
    with av.open(source) as container:
        if not container.streams.video:
            raise MediaError("the file has no video stream")
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        if not frames:
            raise MediaError("the video has no frames")
        stream = container.streams.video[0]
        # torchcodec's average_fps: frames over the duration of the *video stream*, rounded to microseconds (the container is as long as
        # its longest stream, and generated clips often carry an audio track that is a few hundredths of a second longer than the
        # picture; the rounding is why 145 frames in 6.041666... s give 23.99999868 and not 24.0, which decides which frame a
        # window boundary picks)
        duration = round(float(stream.duration * stream.time_base), 6) if stream.duration and stream.time_base else (container.duration or 0) / 1e6
        stream_rate = stream.average_rate
    fps = len(frames) / duration if duration > 0 else float(stream_rate or 25.0)
    if not with_audio:
        return np.stack(frames), fps, None
    if hasattr(source, "seek"):
        source.seek(0)
    chunks, rate = [], None
    with av.open(source) as container:
        if not container.streams.audio:
            raise MediaError("the video has no audio track; connect an AUDIO input or load a video with sound")
        # Keep the encoder-padding samples at the end of the stream (torchcodec returns them, FFmpeg's default decode cuts them) but
        # drop the leading priming samples: skip_manual hands both over as side data and applies neither.
        try:
            container.streams.audio[0].codec_context.options = {"flags2": "+skip_manual"}
            manual = True
        except Exception:  # noqa: BLE001 - an older PyAV: plain decode
            manual = False
        resampler = av.audio.resampler.AudioResampler(format="fltp")
        skip = None
        for frame in container.decode(audio=0):
            rate = frame.sample_rate
            if skip is None:  # the first packet states how many samples to drop in all, which can be more than one frame (2048 in some AAC files)
                skip = _leading_skip(frame) if manual else 0
            chunks.extend(out.to_ndarray() for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray() for out in resampler.resample(None))
    if not chunks:
        raise MediaError("the audio track is empty")
    wave = np.concatenate(chunks, axis=1)
    if skip:
        wave = wave[:, skip:]
    if wave.shape[1] < 1:
        raise MediaError("the audio track is empty")
    return np.stack(frames), fps, {"waveform": torch.from_numpy(wave).unsqueeze(0), "sample_rate": int(rate)}


def load_video(video, with_audio: bool = True):
    """VIDEO -> (frames uint8 [T,H,W,3], fps float, AUDIO dict | None); see the module docstring for the two decode paths."""
    source = _plain_file_source(video)
    if source is not None:
        return decode_file(source, with_audio)
    components = video.get_components()
    images = components.images
    if images is None or images.dim() != 4 or images.shape[0] < 1:
        raise MediaError("the video has no frames")
    frames = (images.detach().float().cpu() * 255.0).round().clamp_(0, 255).to(torch.uint8).numpy()
    return frames, float(components.frame_rate), components.audio


def video_arrays(video, sample_rate: int, audio: dict | None = None):
    """VIDEO (+ optional AUDIO that replaces the video's own track) -> (frames uint8 [T,H,W,3], fps, mono audio at `sample_rate`)."""
    frames, fps, own = load_video(video, with_audio=audio is None)
    track = audio if audio is not None else own
    if track is None:
        raise MediaError("the video has no audio track; connect an AUDIO input or load a video with sound")
    return frames, fps, mono_audio(track, sample_rate)


def shift_waveform(waveform: torch.Tensor, samples: int) -> torch.Tensor:
    """Delay (samples > 0) or advance (samples < 0) the audio and keep its length: zeros fill the gap, the overflow is cut."""
    if samples == 0:
        return waveform
    length = waveform.shape[-1]
    count = min(abs(samples), length)
    gap = torch.zeros(*waveform.shape[:-1], count, dtype=waveform.dtype, device=waveform.device)
    if samples > 0:
        return torch.cat([gap, waveform[..., : length - count]], dim=-1)
    return torch.cat([waveform[..., count:], gap], dim=-1)


def shifted_components(video, offset_ms: float):
    """(images float [T,H,W,3] in 0..1, AUDIO dict, frame rate) of `video` with its audio moved by `offset_ms` (positive = later).

    The frames are the same 8-bit values the scorer reads, so a shifted copy and its original differ only in the audio.
    """
    frames, fps, audio = load_video(video)
    if audio is None:
        raise MediaError("the video has no audio track")
    rate = int(audio["sample_rate"])
    shifted = shift_waveform(audio["waveform"], int(round(float(offset_ms) / 1000.0 * rate)))
    return torch.from_numpy(frames).float().div_(255.0), {"waveform": shifted, "sample_rate": rate}, Fraction(fps)
