"""A tiny stand-in for the PE-AV encoder and processor, so the real SyncReward model code runs on the CPU without weights."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from syncreward_comfy import runtime
from syncreward_comfy.vendor.syncreward.model import PeAVEncoder, SyncRewardModel


class _Tower(nn.Module):
    def __init__(self, in_dim, hidden, tokens, key):
        super().__init__()
        self.proj, self.tokens, self.key = nn.Linear(in_dim, hidden * tokens), tokens, key

    def forward(self, **kw):
        x = kw[self.key].flatten(1)
        return SimpleNamespace(last_hidden_state=self.proj(x).view(x.shape[0], self.tokens, -1))


class _Head(nn.Module):
    def __init__(self, hidden, out):
        super().__init__()
        self.layer_norm, self.proj = nn.LayerNorm(hidden), nn.Linear(hidden, out, bias=False)

    def forward(self, x):
        return self.proj(self.layer_norm(x))


class StubProcessor:
    feature_extractor = SimpleNamespace(sampling_rate=48000)

    def __call__(self, videos, audio, return_tensors, padding, sampling_rate):
        a = torch.stack([torch.from_numpy(x[:64].copy()) for x in audio])
        v = torch.stack([torch.from_numpy(x[:2, :4, :4, :].copy()).permute(0, 3, 1, 2).float() for x in videos])
        return {"input_values": a, "pixel_values_videos": v}


def stub_loaded(seed: int = 0) -> runtime.Loaded:
    torch.manual_seed(seed)
    encoder = PeAVEncoder(_Tower(64, 32, 5, "input_values"), _Tower(96, 32, 3, "pixel_values_videos"), _Head(32, 16), _Head(32, 16))
    model = SyncRewardModel(encoder, num_segments=18, sync_dim=32, num_layers=1, num_heads=2, dropout=0.0).eval()
    segments = {"num_segments": 18, "segment_duration": 0.5, "segment_stride": 0.25, "crop": "center"}
    info = {"checkpoint": "stub", "encoder": "stub", "device": "cpu", "load_seconds": 0.0, "parameters_m": 0.0, "timm_labels_corrected": 0, **segments}
    return runtime.Loaded(key=("stub", "stub", "cpu"), model=model, processor=StubProcessor(), sample_rate=48000, device=torch.device("cpu"), segments=segments, info=info)
