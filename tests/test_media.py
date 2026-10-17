"""VIDEO / AUDIO conversion and the audio shift: no model needed."""

from fractions import Fraction

import numpy as np
import pytest
import torch

from syncreward_comfy import media


def test_shift_delays_and_advances_with_zero_fill_and_same_length():
    wave = torch.arange(1, 11, dtype=torch.float32).reshape(1, 1, 10)
    assert media.shift_waveform(wave, 0) is wave
    assert media.shift_waveform(wave, 3).flatten().tolist() == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
    assert media.shift_waveform(wave, -3).flatten().tolist() == [4, 5, 6, 7, 8, 9, 10, 0, 0, 0]
    assert media.shift_waveform(wave, 50).flatten().tolist() == [0] * 10
    assert media.shift_waveform(wave, -50).flatten().tolist() == [0] * 10
    stereo = torch.randn(1, 2, 100)
    assert media.shift_waveform(stereo, 17).shape == stereo.shape


def test_mono_audio_averages_channels_and_resamples(comfy_dir):
    wave = torch.zeros(1, 2, 24000)
    wave[0, 0] = 0.5
    wave[0, 1] = -0.5
    assert float(np.abs(media.mono_audio({"waveform": wave, "sample_rate": 24000}, 24000)).max()) == 0.0  # channels cancel
    out = media.mono_audio({"waveform": torch.randn(1, 1, 24000), "sample_rate": 24000}, 48000)
    assert out.dtype == np.float32 and out.shape == (48000,)
    with pytest.raises(media.MediaError):
        media.mono_audio({"waveform": torch.zeros(1, 1, 0), "sample_rate": 48000}, 48000)


def test_file_decode_gives_frames_fps_and_audio(sample_video):
    frames, fps, audio = media.decode_file(str(sample_video))
    assert frames.dtype == np.uint8 and frames.shape[1:] == (48, 64, 3) and frames.shape[0] == 30
    assert abs(fps - 25.0) < 0.5
    assert audio["sample_rate"] == 48000 and audio["waveform"].shape[1] == 2 and audio["waveform"].shape[2] >= int(48000 * 1.2) - 2048
    frames_only, _, none_audio = media.decode_file(str(sample_video), with_audio=False)
    assert none_audio is None and np.array_equal(frames_only, frames)


def test_load_video_uses_the_file_for_plain_videos_and_components_otherwise(comfy_dir, sample_video):
    from comfy_api.latest import VideoComponents, VideoFromComponents, VideoFromFile

    plain = VideoFromFile(str(sample_video))
    assert media._plain_file_source(plain) == str(sample_video)
    assert media._plain_file_source(VideoFromFile(str(sample_video), start_time=0.2, duration=0.5)) is None  # trimmed: generic path
    assert media._plain_file_source(VideoFromFile(str(sample_video), crop=(0, 0, 32, 24))) is None

    frames, fps, audio = media.load_video(plain)
    built = VideoFromComponents(VideoComponents(images=torch.from_numpy(frames).float() / 255.0, audio=audio, frame_rate=Fraction(fps)))
    frames2, fps2, audio2 = media.load_video(built)
    assert np.array_equal(frames, frames2) and fps == fps2  # uint8 -> float -> uint8 is exact
    assert torch.equal(audio["waveform"], audio2["waveform"])


def test_video_arrays_prefers_an_audio_override_and_reports_missing_audio(comfy_dir, sample_video):
    from comfy_api.latest import VideoComponents, VideoFromComponents, VideoFromFile

    video = VideoFromFile(str(sample_video))
    frames, fps, wave = media.video_arrays(video, 48000)
    assert wave.dtype == np.float32 and wave.ndim == 1 and frames.shape[0] == 30
    override = {"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000}
    _, _, silent = media.video_arrays(video, 48000, override)
    assert float(np.abs(silent).max()) == 0.0
    no_audio = VideoFromComponents(VideoComponents(images=torch.zeros(3, 8, 8, 3), frame_rate=Fraction(25)))
    with pytest.raises(media.MediaError, match="no audio track"):
        media.video_arrays(no_audio, 48000)
    _, _, taken = media.video_arrays(no_audio, 48000, override)  # an AUDIO input makes such a video scorable
    assert taken.shape == (48000,)


def test_shifted_components_keep_the_exact_frames_and_move_only_the_audio(comfy_dir, sample_video):
    from comfy_api.latest import VideoFromFile

    video = VideoFromFile(str(sample_video))
    frames, _, audio = media.load_video(video)
    images, shifted, rate = media.shifted_components(video, 100.0)
    assert (images * 255).round().to(torch.uint8).numpy().tobytes() == frames.tobytes()
    count = int(round(0.1 * 48000))
    assert shifted["waveform"].shape == audio["waveform"].shape and shifted["sample_rate"] == 48000
    assert torch.equal(shifted["waveform"][..., count:], audio["waveform"][..., :-count])
    assert float(shifted["waveform"][..., :count].abs().max()) == 0.0
    assert float(rate) > 0
