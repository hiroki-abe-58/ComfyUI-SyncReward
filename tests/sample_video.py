"""A tiny MP4 with a stereo AAC track made with PyAV, so no media files are stored in the repository."""

from pathlib import Path


def make_sample_video(path: Path, fps: int = 25, seconds: float = 1.2, rate: int = 48000) -> Path:
    import av
    import numpy as np

    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=fps)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = container.add_stream("aac", rate=rate)
        audio.layout = "stereo"
        for i in range(int(fps * seconds)):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), (i * 8) % 255, dtype=np.uint8), format="rgb24")
            for packet in video.encode(frame.reformat(format="yuv420p")):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
        t = np.arange(int(rate * seconds)) / rate
        wave = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(np.stack([wave, wave])), format="fltp", layout="stereo")
        frame.sample_rate = rate
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode(None):
            container.mux(packet)
    return path
