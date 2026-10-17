"""ComfyUI nodes for SyncReward (in-process; no subprocess, no download at run time)."""

from __future__ import annotations

import json

from comfy_api.latest import ComfyExtension, VideoComponents, VideoFromComponents, io

from . import media, paths, runtime

SyncRewardModel = io.Custom("SYNCREWARD_MODEL")

SCORE_HELP = (
    "A reward between 0 (audio and picture are not synchronized) and 2 (well synchronized), predicted by the model; it was trained on "
    "human synchronization ratings of generated clips. It scores a centered 4.75 s window (18 overlapping 0.5 s segments); for shorter "
    "clips the last frame / silence is repeated."
)


def _score_clip(loaded, video, audio, batch_size):
    frames, fps, wave = media.video_arrays(video, loaded.sample_rate, audio)
    return runtime.score(loaded, frames, fps, wave, batch_size)


def _run(model, work):
    """Run `work(loaded)`; free the model afterwards when the Loader asked for it."""
    loaded = runtime.acquire(model)
    info = loaded.info
    try:
        return work(loaded), info
    finally:
        del loaded
        if not model.get("keep_loaded", True):
            runtime.release()


class SyncRewardLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyncRewardLoader",
            display_name="SyncReward Loader",
            category="loaders/syncreward",
            description=(
                "Loads the SyncReward Stage-2 reward model (about 1.66 B parameters, fp32 on the GPU as in the official code) together "
                "with the PE-AV processor from models/syncreward. One model stays resident; loading another frees it. The weights are "
                "not included in this node pack."
            ),
            inputs=[
                io.Combo.Input("checkpoint", options=paths.checkpoints(), tooltip="The syncreward-stage2 folder of Guan123/SyncReward in models/syncreward"),
                io.Combo.Input("encoder", options=paths.encoders(), tooltip="The facebook/pe-av-large folder in models/syncreward"),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"], default="auto"),
                io.Boolean.Input("keep_loaded", default=True, tooltip="Keep the model in memory after the job. Turn off to free VRAM for other models after every run."),
            ],
            outputs=[SyncRewardModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, checkpoint, encoder, device, keep_loaded) -> io.NodeOutput:
        handle = runtime.make_handle(paths.checkpoint_path(checkpoint), paths.encoder_path(encoder), device, keep_loaded)
        return io.NodeOutput(handle)


class SyncRewardScore(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyncRewardScore",
            display_name="SyncReward Score",
            category="video/syncreward",
            description="Audio-visual synchronization score of a video with sound. " + SCORE_HELP,
            inputs=[
                SyncRewardModel.Input("model"),
                io.Video.Input("video", tooltip="A video that has an audio track (for example from Load Video)"),
                io.Audio.Input("audio", optional=True, tooltip="Optional: score this audio against the video's picture instead of the video's own track"),
                io.Int.Input("encode_batch_size", default=18, min=1, max=36,
                             tooltip="Segments encoded at once (official default 18). Lower it to save VRAM (9.8 GiB at 18, 6.7 GiB at 1); the reward "
                                     "can then move by one bfloat16 step (0.008) for some clips."),
            ],
            outputs=[io.Float.Output(display_name="reward"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, video, encode_batch_size, audio=None) -> io.NodeOutput:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
        report, info = _run(model, lambda loaded: _score_clip(loaded, video, audio, encode_batch_size))
        return io.NodeOutput(report["reward"], runtime.report_json(info, report))


class SyncRewardSelectBest(io.ComfyNode):
    MAX_VIDEOS = 6

    @classmethod
    def define_schema(cls):
        optional = [io.Video.Input(f"video_{i}", optional=True) for i in range(3, cls.MAX_VIDEOS + 1)]
        return io.Schema(
            node_id="SyncRewardSelectBest",
            display_name="SyncReward Select Best",
            category="video/syncreward",
            description=(
                "Scores two to six candidate videos (for example several generations of the same prompt) one after another and returns the "
                "one with the highest synchronization reward, its number (1 = video_1), its reward and every score. Ties go to the lower "
                "number. " + SCORE_HELP
            ),
            inputs=[
                SyncRewardModel.Input("model"),
                io.Video.Input("video_1"),
                io.Video.Input("video_2"),
                *optional,
                io.Int.Input("encode_batch_size", default=18, min=1, max=36),
            ],
            outputs=[
                io.Video.Output(display_name="best_video"),
                io.Int.Output(display_name="best_index"),
                io.Float.Output(display_name="best_reward"),
                io.String.Output(display_name="scores"),
            ],
        )

    @classmethod
    def execute(cls, model, video_1, video_2, encode_batch_size, **optional) -> io.NodeOutput:
        import comfy.model_management as mm
        import comfy.utils

        candidates = [(1, video_1), (2, video_2)]
        candidates += [(i, optional[f"video_{i}"]) for i in range(3, cls.MAX_VIDEOS + 1) if optional.get(f"video_{i}") is not None]
        bar = comfy.utils.ProgressBar(len(candidates))

        def work(loaded):
            results = []
            for number, video in candidates:
                mm.throw_exception_if_processing_interrupted()
                results.append((number, video, _score_clip(loaded, video, None, encode_batch_size)))
                bar.update(1)
            return results

        results, info = _run(model, work)
        best_number, best_video, best_report = max(results, key=lambda item: (item[2]["reward"], -item[0]))
        scores = {
            "model": {k: info[k] for k in ("checkpoint", "encoder", "device")},
            "best": best_number,
            "rewards": {f"video_{n}": round(r["reward"], 4) for n, _, r in results},
        }
        return io.NodeOutput(best_video, best_number, best_report["reward"], json.dumps(scores, indent=1))


class SyncRewardShiftAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyncRewardShiftAudio",
            display_name="SyncReward Shift Audio",
            category="video/syncreward",
            description=(
                "Moves a video's audio in time without touching the picture or the length: a positive offset delays the sound (silence "
                "fills the gap, the end is cut), a negative one advances it. Use it to make deliberately out-of-sync test clips."
            ),
            inputs=[
                io.Video.Input("video", tooltip="A video that has an audio track"),
                io.Float.Input("offset_ms", default=250.0, min=-5000.0, max=5000.0, step=10.0, tooltip="Positive = audio later than the picture"),
            ],
            outputs=[io.Video.Output(display_name="video")],
        )

    @classmethod
    def execute(cls, video, offset_ms) -> io.NodeOutput:
        images, audio, frame_rate = media.shifted_components(video, offset_ms)
        return io.NodeOutput(VideoFromComponents(VideoComponents(images=images, audio=audio, frame_rate=frame_rate)))


class SyncRewardUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyncRewardUnload",
            display_name="SyncReward Unload",
            category="loaders/syncreward",
            description="Frees the resident SyncReward model (GPU and RAM).",
            inputs=[io.Float.Input("after", optional=True, force_input=True, tooltip="Optional: connect a reward so this runs after it")],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, after=None) -> io.NodeOutput:
        import torch

        was = runtime._SLOT is not None
        runtime.release()
        text = json.dumps({"unloaded": was, "cuda_allocated_mib": torch.cuda.memory_allocated() // 2**20 if torch.cuda.is_available() else None}, indent=1)
        return io.NodeOutput(text, ui={"text": [text]})


class SyncRewardExtension(ComfyExtension):
    async def get_node_list(self):
        return [SyncRewardLoader, SyncRewardScore, SyncRewardSelectBest, SyncRewardShiftAudio, SyncRewardUnload]


async def comfy_entrypoint() -> SyncRewardExtension:
    paths.register()
    return SyncRewardExtension()
