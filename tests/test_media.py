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


def test_frame_rate_comes_from_the_video_stream_not_from_a_longer_audio_track(tmp_path):
    from sample_video import make_sample_video

    path = make_sample_video(tmp_path / "long_audio.mp4", seconds=1.2, audio_seconds=1.8)  # container is 1.8 s, the picture 1.2 s
    frames, fps, audio = media.decode_file(str(path))
    assert frames.shape[0] == 30 and abs(fps - 25.0) < 0.3  # frames / container duration would be 16.7
    assert audio["waveform"].shape[2] > int(48000 * 1.7)


def test_frame_rate_uses_the_stream_duration_rounded_to_microseconds_like_torchcodec(tmp_path):
    from sample_video import make_sample_video

    path = make_sample_video(tmp_path / "ntsc_like.mp4", fps=24, seconds=145 / 24)  # 145 frames of 1/24 s: 6.0416666... s
    frames, fps, _ = media.decode_file(str(path), with_audio=False)
    assert frames.shape[0] == 145
    assert fps == 145 / 6.041667 and fps != 24.0  # torchcodec reports 23.99999867586214 here; the window edges (k * 0.25 s * fps) depend on it


def test_leading_skip_of_the_first_packet_is_dropped_in_all_even_beyond_one_frame(sample_video, monkeypatch):
    monkeypatch.setattr(media, "_leading_skip", lambda frame: 1024)  # one frame's worth, what most AAC files state
    _, _, one_frame = media.decode_file(str(sample_video))
    monkeypatch.setattr(media, "_leading_skip", lambda frame: 2048)  # some AAC files (for example kling clips) state 2048 on a first frame that holds 1024
    _, _, two_frames = media.decode_file(str(sample_video))
    assert one_frame["waveform"].shape[2] - two_frames["waveform"].shape[2] == 1024
    assert torch.equal(two_frames["waveform"], one_frame["waveform"][..., 1024:])


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
