# SyncReward for ComfyUI

Score how well the **sound of a video is synchronized with its picture**, and pick the best of several generated videos, in ComfyUI
with the official **SyncReward** model ([WeChatCV/SyncReward](https://github.com/WeChatCV/SyncReward), paper
[arXiv:2610.16338](https://arxiv.org/abs/2610.16338)). The model code runs **in-process** (no subprocess, no server, no download at run
time) on ComfyUI's standard `VIDEO` and `AUDIO` types.

> **Unofficial.** This node pack is an independent integration. It is not affiliated with or endorsed by the SyncReward authors
> (Renmin University of China, WeChat / Tencent), by Meta (PE-AV) or by Comfy Org.
>
> 日本語の概要は [README.ja.md](README.ja.md) にあります。

## What it does

| Node | Purpose |
| --- | --- |
| **SyncReward Loader** | Picks the Stage-2 checkpoint and the PE-AV encoder folder from `models/syncreward`, the device and a keep-loaded switch |
| **SyncReward Score** | `VIDEO` (with sound, or plus an `AUDIO` that replaces the track) → reward `FLOAT` between 0 and 2 and a JSON report |
| **SyncReward Select Best** | Two to six `VIDEO` candidates → the one with the highest reward, its number, its reward and every score |
| **SyncReward Shift Audio** | Moves a video's audio in time (picture and length unchanged): for deliberately out-of-sync test clips |
| **SyncReward Unload** | Frees the resident model (GPU and RAM) |

The reward is the model's prediction of how well audible events line up with visible actions (0 = not synchronized, 2 = well
synchronized). It was trained on human ratings of *generated* audio-video clips. It looks at a centered **4.75 s window** of the clip
(18 overlapping 0.5 s segments; for shorter clips the last frame / silence is repeated).

## How far it was checked

Windows 11, RTX 5090 (32 GB), ComfyUI v0.38.0, the ten demo clips of the official repository (details and raw numbers in
[docs/VERIFICATION.md](docs/VERIFICATION.md)):

- **Same decoding and same rewards as the official code.** In one Python environment (torch 2.9.1, transformers 5.1.0, the official
  requirements), the frames, the frame rate and the 48 kHz mono audio decoded by this pack equal what the official `load_av` returns,
  bit for bit, for 10 of 10 clips, and the rewards from the official model code and from this pack are identical for 10 of 10.
- **Runs through the real ComfyUI queue** in a ComfyUI environment with torch 2.14.1 and transformers 5.20.1, from a clean install:
  9 of 10 rewards equal the official command-line result; one differs by a single bfloat16 step (0.008), which is the library
  version (the same arrays give the same value in both environments for 9 of 10 clips and 1.109 vs 1.117 for the tenth).
- **Real jobs:** score, select-best, shift-and-score, error messages, unload, keep-loaded off, CPU device, all with the real weights.
- CI (GitHub Actions, Ubuntu and Windows, transformers 5.1.0 and latest): CPU tests with a stub encoder, and a real ComfyUI queue with
  the model-free nodes. CI does not download the weights (about 15 GB).

- **The paper's benchmark is reproduced** (2026-10-18): all 1,000 clips of SyncReward-Bench scored by this pack give Spearman 0.781,
  Top-1 (N = 4) 0.764 and pairwise accuracy 0.835 against the model card's 0.7823 / 0.7646 / 0.8359 (inside the 95 % intervals; the
  official `evaluate.py` run on the same machine gives 0.7810 / 0.7641 / 0.8351). In the official environment the rewards of the pack and of
  `evaluate.py` differ by at most one bfloat16 step (0.008) on all 1,000 clips. See [LIMITATIONS.md](LIMITATIONS.md).

**Not checked:** Linux or macOS with a GPU, AMD / Apple GPUs, other ComfyUI versions with the real model (the CPU tests and the real-queue check also run against ComfyUI v0.41.0 in CI), real (not generated) videos, clips that are not
about 5 s long, and the case *Select Best* is meant for: several candidates of the same prompt from the same generator.

> **Read [LIMITATIONS.md](LIMITATIONS.md) before you rely on the score or on *Select Best*.** Measured on 70 well-synchronized
> benchmark clips: moving the audio by up to ±1 s changes the reward by 0.01 on average and is not detected; silence, noise or another
> clip's sound lowers it clearly in only about 30 to 70 % of the clips; and a heavy re-encode, half resolution or 16 kHz audio lowers it
> by 55 to 80 % as much as removing the sound does. The score agrees with human raters on average (benchmark above) but it is **not a timing detector** and
> not a test for missing sound.

## Install

1. ComfyUI with the V3 node API (**v0.38.0** was used), Python 3.10+, an NVIDIA GPU with about 10 GB of free VRAM (CPU works, slowly).
2. Put this folder into `ComfyUI/custom_nodes/` (or install `syncreward` from the ComfyUI Manager / Comfy Registry) and install its
   requirements into ComfyUI's Python:

   ```
   python -m pip install -r custom_nodes/ComfyUI-SyncReward/requirements.txt
   ```

   That is `transformers>=5.1.0` (PE-AV exists only there), `timm` and `PyYAML`. The pack works with transformers 5.1.0 and 5.20.1
   (see *Notes*).
3. Download the two models (about 15 GB together; both model cards say Apache-2.0) into `models/syncreward/`:

   ```
   python custom_nodes/ComfyUI-SyncReward/tools/download_models.py --models-dir ComfyUI/models/syncreward
   ```

   This fetches `syncreward-stage2` (Guan123/SyncReward) and `pe-av-large` (facebook/pe-av-large) at the revisions this pack was
   verified with and checks their SHA-256. The loader lists a folder only if it holds the right files (`config.json` with
   `"model_type": "syncreward"` / `"pe_audio_video"`, `model.safetensors`, and for the encoder the processor files).
4. Restart ComfyUI and load a workflow from `workflows/gui/` (or `workflows/api/`). Their Load Video node points at `example.mp4`:
   choose one of your own videos there.

Nothing is downloaded or installed by the nodes themselves.

## Using the nodes

- **Score**: connect a *Load Video* to **video**. Videos without a sound track need an **audio** input. The report shows the reward,
  the clip length, whether the clip was shorter than the 4.75 s window and the time and peak VRAM of that run.
- **Select Best**: connect two to six videos (for example several generations of the same prompt); wire **best_video** to *Save Video*.
  Ties go to the lower number. The scores are listed in the JSON output. **Treat the choice as a suggestion.** On benchmark clips
  that humans rated close together (gap up to a third of a rating step) the reward orders a pair like the humans only 69 % of the time;
  rewards that differ by less than about 0.1 are noise; candidates that were encoded differently (resolution, bitrate, frame rate) are
  partly chosen by that; and a candidate with no sound, noise or someone else's sound can still win ([LIMITATIONS.md](LIMITATIONS.md)).
- **Shift Audio**: `offset_ms` > 0 delays the sound, < 0 advances it. The picture and the length stay the same.
- **Decoding.** An unmodified *Load Video* is read from its file the way the official code does (plain 8-bit RGB frames, frame rate =
  frames / duration of the video stream, all audio samples after the leading priming). ComfyUI's own decoder converts untagged videos with a different colour matrix,
  which changes pixels by up to about 40 of 255 and the reward by up to 0.07 on the demo clips, so it is not used for them. Trimmed,
  cropped, image-built or otherwise processed videos are converted from ComfyUI's decoded frames instead.
- **encode_batch_size** (default 18, as in the official code) trades VRAM for nothing else: 9.8 GiB at 18, 6.7 GiB at 1; the reward can
  move by one bfloat16 step (0.008) for some clips.

## What the score reacts to (measured on the 10 demo clips)

The official repository ships five before / after pairs (the "after" clips come from the reward-guided post-training). Rewards from
this pack:

| pair | before | after |
| --- | --- | --- |
| case 1 | 0.062 | 1.789 |
| case 2 | 0.609 | 1.727 |
| case 3 | 1.031 | 1.148 |
| case 4 | 0.727 | 1.234 |
| case 5 | 0.797 | 1.109 |

The "after" clip scores higher in all five pairs, which is what the authors report. Moving the audio of the same ten clips
(*Shift Audio*; positive = later; mean reward over the ten clips, unshifted 1.023):

| offset | -1000 ms | -500 | -250 | -100 | +100 | +250 | +500 | +1000 | +2000 | +3000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean reward | 0.993 | 1.017 | 1.024 | 1.033 | 1.023 | 1.018 | 1.009 | 0.991 | 0.951 | 0.879 |

and replacing the audio: silence 0.718, the audio of another clip 0.824, white noise of the same loudness 0.987, the audio played
backwards 1.031. So **in this setup the reward is a coarse measure**: it drops clearly for missing or unrelated sound and by a few
hundredths for offsets of up to one second, and the unshifted clip had the highest reward in only 4 of 10 clips in the offset sweep. Do not use
it to detect small timing offsets or reversed audio. (Single runs on one machine; the sweep and the controls are
`scripts/experiment_audio_offset.py` and `scripts/experiment_controls.py`. A larger set, 70 well-synchronized benchmark clips with
shifts of 0.2 / 0.5 / 1.0 s, reversed, silent, noisy and unrelated sound and six re-encodings, gives the same picture:
[LIMITATIONS.md](LIMITATIONS.md).)

## Measured on this setup

RTX 5090, Windows 11, torch 2.14.1+cu130, bfloat16 autocast as in the official `score.py`: loading takes about 10 s (the weights
stay fp32 on the GPU: 1.66 B parameters), scoring a 5 s clip about 1.6 to 2 s, peak VRAM allocated by the model 9.8 GiB (6.7 GiB with
`encode_batch_size` 1). On the CPU (fp32) the same clip took 103 s including loading. System commit charge rose to about 37 to 39 GiB
while loading (the PE-AV checkpoint is read once as a whole before the Stage-2 weights replace its towers).

## Notes and limits

- **transformers 5.20 and the PE-AV vision head.** transformers 5.20.1 reads the nested timm configuration of
  `facebook/pe-av-large` with 2 labels instead of 1024 (5.1.0, which the official repository pins, reads 1024), so the vision head and
  projection get the wrong shape and loading fails. The pack restores `num_classes` from the model's `config.json` before building the
  encoder; the rest of the loading is the official `build_reward_model` code. A log line `padding_mask_videos is part of ...
  get_video_features's signature, but not documented` appears while loading: it comes from transformers and is harmless.
- **Windows and the official command line.** On this machine the official `score.py` ran with torch 2.9.1 (cu128), torchaudio 2.9.1,
  torchcodec 0.8.1 and FFmpeg 7 shared DLLs made visible with `os.add_dll_directory`; with torch/torchaudio 2.8 on Windows,
  `torchaudio.load` of an `.mp4` failed ("Couldn't find appropriate backend"). This pack does not use torchaudio or torchcodec.
- The Stage-2 checkpoint's own `config.json` supplies the window (18 x 0.5 s, stride 0.25 s) and `crop = center`; the other crop modes
  of the official CLI are training augmentations and are not offered.
- The reward is a prediction from a model trained on generated videos; it is not a ground-truth measurement of timing.

## Licenses

- This node pack's code: **MIT** ([LICENSE](LICENSE)).
- `syncreward_comfy/vendor/syncreward`: copy of the official package, **Apache-2.0** ([NOTICE](NOTICE),
  `syncreward_comfy/vendor/LICENSE`); one training-only function in `utils.py` is disabled (see `syncreward_comfy/vendor/UPSTREAM.md`),
  everything else is unmodified.
- Model weights are **not included**: SyncReward Stage 2 and Meta's PE-AV encoder, both Apache-2.0 according to their model cards.
- The demo videos of the official repository are **not included** here and are not covered by these licenses; the numbers above were
  measured on them.
- ComfyUI (GPL-3.0) hosts the nodes; PyTorch, transformers, timm and PyAV keep their own licenses.

## Development

```
pip install pytest ruff                                   # plus ComfyUI's requirements
SYNCREWARD_COMFYUI_DIR=... SYNCREWARD_UPSTREAM_DIR=... SYNCREWARD_PEAV_CONFIG=... python -m pytest tests
python scripts/ci_queue_check.py --comfyui <ComfyUI>      # real queue on CPU, model-free nodes
python scripts/parity_vs_official.py --upstream <SyncReward checkout> --checkpoint <stage2 folder> --encoder <pe-av folder> clip.mp4 ...
python scripts/queue_scores.py --server http://127.0.0.1:8188 --comfy-input <ComfyUI/input> --official official.jsonl clip.mp4 ...
```

`scripts/queue_client.py` submits any API workflow to a running ComfyUI. Refreshing the vendored code: replace
`syncreward_comfy/vendor/syncreward` with the new upstream files, run `scripts/vendor_patch.py --upstream <checkout>` and
`scripts/update_vendor_manifest.py`, update `UPSTREAM.md` and
the commit in `.github/workflows/ci.yml`.

## Citation

If you use SyncReward, please cite the authors (see the upstream README for the BibTeX of arXiv:2610.16338).
