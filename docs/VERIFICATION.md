# Verification record

What was run, on what, and what came out. Everything here was run on 2026-10-17/18 on one machine; numbers are single runs unless
stated. The ten clips are the demo videos of the official repository (`assets/demo/case{1..5}-{before,after}.mp4`, 5 s, 768x512, 24 fps,
stereo AAC at 48 kHz); they are read locally and not redistributed.

## Environment

| | |
| --- | --- |
| OS / GPU | Windows 11 Pro (10.0.26200), NVIDIA GeForce RTX 5090 32 GB (compute capability 12.0), driver 610.88, 64 GB RAM |
| ComfyUI | v0.38.0 (`6b747c0428c343e1417219641db93a4fb7cb69ae`), Python 3.12.13, own virtual environment, 127.0.0.1 only |
| ComfyUI environment | torch 2.14.1+cu130, transformers 5.20.1, timm 1.0.26, av 19.0.1 |
| Official-reference environment | torch 2.9.1+cu128, torchaudio 2.9.1, torchcodec 0.8.1, transformers 5.1.0, timm 1.0.26, PyAV 19.0.1 (only for this pack's decoder in the comparison), FFmpeg 7.1.1 shared DLLs registered with `os.add_dll_directory` |
| Upstream code | WeChatCV/SyncReward `f29fbdc2bb2dd3d59644ffd3c60d207dd3e62615` (vendored unmodified, checked by `tests/test_vendor.py`) |
| Models | Guan123/SyncReward `04bc7701da8d8466c7b3aa2dec22a2a5051e6f0a`, folder `syncreward-stage2`; facebook/pe-av-large `0d24878d4107d64bef49e53602fc34ce6f94f6d8` (both: model card Apache-2.0) |

Weight file SHA-256 (equal to the values published by the Hub):

```
6c09263ac41d59baf78a181f17e4116128a337d70a03637886de3734450a32a2  syncreward-stage2/model.safetensors   (fp32, 6.18 GB)
0bb0e1e8b5d236352e1a828a89cd7a97af8c8f9b6dc43312f58e49d0faa71651  syncreward-stage2/config.json
7192bdcfc5fec52687b36c92db404a303b0d9bbbb85d26839e530b4f3d44a354  pe-av-large/model.safetensors        (8.94 GB)
```

## 1. The official code on this machine

The official `score.py`, unmodified, ran on the ten clips with the Stage-2 checkpoint in the official-reference environment (torch 2.9.1).
Getting there on Windows needed three things the official requirements (torch 2.6, torchaudio 2.6, FFmpeg 4.4.2 on H20 GPUs) do not
say: a PyTorch build that supports compute capability 12.0 (2.9.1+cu128 here), `torchaudio.load` of `.mp4` working (it raised
"Couldn't find appropriate backend" with torchaudio 2.8 on Windows; 2.9.1 delegates to torchcodec), and FFmpeg's shared DLLs made visible
to torchcodec (`os.add_dll_directory`). 23 s for the ten clips including loading; the system GPU memory in use peaked at 15,715 MiB (15.3 GiB, including the desktop).

| clip | before | after |
| --- | --- | --- |
| case 1 | 0.0623 | 1.7891 |
| case 2 | 0.6094 | 1.7266 |
| case 3 | 1.0312 | 1.1484 |
| case 4 | 0.7266 | 1.2344 |
| case 5 | 0.7969 | 1.1172 |

## 2. This pack against the official code, clip by clip

`scripts/parity_vs_official.py`, both in the official-reference environment, each with its own model instance:

| | frames identical | frame rate identical | 48 kHz mono audio identical | reward identical |
| --- | --- | --- | --- | --- |
| 10 clips | 10 / 10 | 10 / 10 | 10 / 10 | **10 / 10** |

How it got there (the decoder was changed twice during development, so this is worth knowing):
- ComfyUI's own decoder returns different pixels than torchcodec for videos without colour tags (it uses a BT.709 matrix, torchcodec /
  plain PyAV the default): mean absolute difference 1.1 of 255, up to 42; the reward differed (case 1 after: 1.719 vs 1.789). The pack
  therefore decodes an unmodified `Load Video` from its file with PyAV's plain `rgb24` conversion, which reproduced torchcodec's frames
  exactly.
- The audio: FFmpeg's default decode cuts the encoder padding at the end of the AAC stream (240,480 samples) while torchcodec returns it
  (240,640): the 160 extra samples move the centered window by 1.7 ms and flipped the reward of some clips by one bfloat16 step. The
  pack decodes with `flags2=+skip_manual`, drops the leading priming samples only, and then equals torchcodec sample for sample.
- The frame rate is frames / duration of the video stream, as in torchcodec (`23.999998413...`, not 24). (0.1.0 divided by the *container*
  duration and dropped the priming samples of the first audio *frame* only. The ten demo clips cannot show the difference; the
  1,000 benchmark clips did, see section 10.)

## 3. Through the real ComfyUI queue (ComfyUI environment, torch 2.14.1, transformers 5.20.1)

A fresh ComfyUI v0.38.0 with this pack installed and the models in `models/syncreward`; `scripts/queue_scores.py` posts the shipped
`score_basic` workflow (Load Video -> SyncReward Score) for each clip through `POST /prompt`:

| clip | pack | official | difference |
| --- | --- | --- | --- |
| case1-before | 0.06226 | 0.06226 | 0 |
| case1-after | 1.78906 | 1.78906 | 0 |
| case2-before | 0.60938 | 0.60938 | 0 |
| case2-after | 1.72656 | 1.72656 | 0 |
| case3-before | 1.03125 | 1.03125 | 0 |
| case3-after | 1.14844 | 1.14844 | 0 |
| case4-before | 0.72656 | 0.72656 | 0 |
| case4-after | 1.23438 | 1.23438 | 0 |
| case5-before | 0.79688 | 0.79688 | 0 |
| case5-after | 1.10938 | 1.11719 | -0.0078 |

9 of 10 identical; the tenth is one bfloat16 step away. The cause is the library versions, not the decoding: for `case5-after` the
pack's arrays are bit-identical to the official ones, and the official arrays also give 1.10938 in the ComfyUI environment.
Per scoring call: 1.5 to 2.0 s; peak VRAM allocated by the model 9.816 GiB at `encode_batch_size` 18.

`encode_batch_size` (clips case2-after / case3-before / case5-after): rewards 1.7266 / 1.0312 / 1.1094 at batch sizes 18, 9, 6 and 1;
at batch size 3 two of the three moved up by 0.0078. Peak VRAM allocated: 9.82 / 8.16 / 7.60 / 7.05 / 6.68 GiB at 18 / 9 / 6 / 3 / 1.

## 4. A compatibility finding: transformers 5.20 and the PE-AV vision head

With transformers 5.20.1 the official `build_reward_model` fails on the PE-AV checkpoint:
`video_model.video_encoder.embedder.vision_model.timm_model.head.weight | MISMATCH | ckpt torch.Size([1024, 1024]) vs model torch.Size([2, 1024])`.
The nested `timm_wrapper` configuration (`config.audio_video_config.video_config.vision_config`) is read with `num_labels = 2` instead of the
`num_classes = 1024` of `config.json`; transformers 5.1.0 reads 1024. (timm 1.0.26 and 1.0.30 behave the same; the version of timm is
not the cause.) The pack restores `num_classes` on that sub-configuration before building the encoder and then runs the official code
(`tests/test_runtime_stub.py::test_pe_av_config_gets_its_vision_head_width_back` checks it against the real `config.json`). With
transformers 5.1.0 nothing is changed (the pack reports `timm_labels_corrected: 0`).

**Minimal reproduction and cause (2026-10-18).** `python scripts/repro_timm_labels.py <folder with config.json>` prints the value
for the installed version (5.1.0: 1024, 5.20.1: 2). Without any file: `PeVideoEncoderConfig().vision_config.num_labels` is 1024 in 5.1.0 and
2 in 5.20.1; `TimmWrapperConfig.from_dict({"num_classes": 1024, ...}).num_labels` is 1024 in both, `TimmWrapperConfig(num_classes=1024, ...)` is 2 in
both. In 5.20.1 the nested configuration is created by `SubConfigSpec.create_subconfig` as `subconfig_cls(**subconfig)`, which skips
`TimmWrapperConfig.from_dict` (the only place that turns `num_classes` into `num_labels`); in 5.1.0 `PeVideoEncoderConfig` called
`from_dict`. The same code is on transformers `main` (d4804f9, 2026-10-17); 5.20.1 is the latest release. A fix that makes
`TimmWrapperConfig` accept `num_classes` itself restores 1024 (tested by patching `__post_init__` in 5.20.1: default config, the real
`config.json` and ordinary `num_labels` all behave); I did not run the transformers test suite. No issue or PR was opened in the
transformers repository; a draft exists locally.

## 5. Behavior with the real weights (queue)

| Check | Result |
| --- | --- |
| Score a clip | 11.2 s wall for the first job (loading ~9.8 s); later jobs: scoring 1.5 to 2.0 s plus decoding |
| Shift Audio 250 ms then Score vs the original (case3-after) | 1.1484 vs 1.1484 |
| Select Best over case3-before / case3-after / case5-before | best = 2 (1.1484); rewards 1.0312 / 1.1484 / 0.7969 |
| Select Best over +500 ms / original / -250 ms of case2-after | all 1.7266 -> number 1 (ties go to the lower number) |
| Checkpoint folder that does not exist | rejected by ComfyUI's validation before queuing |
| `keep_loaded = false` | GPU memory in use fell from 8470 MiB to 2134 MiB after the job |
| SyncReward Unload node | 8470 MiB to 2134 MiB |
| CPU device (fp32, real weights) | succeeded: reward 1.7917 (GPU bf16: 1.7891) in 103 s including loading |
| System commit charge | 19 to 39 GiB (limit 111.4 GiB) during the runs; lowest free RAM 35 GiB |

## 6. What the reward reacts to (10 clips, `scripts/experiment_audio_offset.py`, `scripts/experiment_controls.py`)

Mean reward over the ten clips; unshifted 1.0234. Positive offsets delay the sound; the picture and the length do not change.

| offset | -1000 ms | -500 | -250 | -100 | +100 | +250 | +500 | +1000 | +2000 | +3000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean reward | 0.9933 | 1.0166 | 1.0242 | 1.0325 | 1.0228 | 1.0175 | 1.0088 | 0.9905 | 0.9507 | 0.8790 |

| other audio | silence | audio of another clip | white noise, same RMS | played backwards |
| --- | --- | --- | --- | --- |
| mean reward | 0.7176 | 0.8237 | 0.9869 | 1.0311 |

In the offset sweep (9 offsets, 10 clips) the unshifted clip had the highest reward in 4 of 10 clips and 57.5 % of the shifted runs were
below their unshifted run. Per-clip rewards are in the raw files written by the scripts. Reading: the reward drops clearly for missing or
unrelated sound and by a few hundredths for offsets up to 1 s; it did not react to time reversal. This says nothing about agreement
with human raters (the paper's 1,000-clip benchmark was not run here).

## 7. Tests

| Where | What | Result |
| --- | --- | --- |
| local, torch 2.14.1 CPU, transformers 5.20.1 | ruff; `pytest tests` (video/audio conversion and decoding, the audio shift, scoring with a stub encoder, the PE-AV config fix against the real `config.json`, vendored-code manifest + byte comparison with the pinned commit, node registration, ComfyUI validation of the shipped workflows, node execution) | 26 passed |
| local, `scripts/ci_queue_check.py` | a second fresh ComfyUI checkout with the pack copied in, `--cpu`: object_info, then Load Video -> Shift Audio -> Get Video Components through `POST /prompt` | passed |
| ComfyUI frontend | `scripts/export_gui_workflows.py` loads each API workflow with the frontend's `loadApiJson`: no missing node types, `graphToPrompt` returns the same classes and values, no page errors | 3 of 3 |

GitHub Actions results are linked from the repository's release notes.

## 8. Not verified

Linux or macOS with a GPU; AMD, Intel or Apple GPUs; other ComfyUI, PyTorch or transformers versions than the ones above; real (not generated)
videos; candidates of the same prompt from the same generator; clips at frame rates other than about 24 fps or lengths other than about 5 s;
videos whose audio is not AAC; any comparison with other synchronization metrics. (The paper's human-rated benchmark and the controlled
changes are in section 9.)

## 9. The human-rated benchmark and controlled changes (2026-10-18, `scripts/eval_suite.py`)

Same environment as section 3 (torch 2.14.1, transformers 5.20.1, bfloat16 autocast), in-process through the pack's own decoding and scoring
code (version 0.1.1, section 10). The 1,000 clips of SyncReward-Bench (dataset revision `a7904b6d5901`, 3.16 GB, fetched clip by clip from the
dataset's zip files with `scripts/fetch_bench_clips.py`; kept locally, not redistributed) were scored in about 35 minutes (1.3 to 2.5 s per clip);
0 failures.

| metric | this pack (ComfyUI environment) | 95 % bootstrap interval | official `evaluate.py`, run here | model card |
| --- | --- | --- | --- | --- |
| Spearman | 0.7810 | [0.756, 0.805] | 0.7810 | 0.7823 |
| Top-1 (N = 4, 1000 groupings; chance 0.3323) | 0.7639 | [0.713, 0.821] | 0.7641 | 0.7646 |
| Pairwise accuracy (421,748 pairs) | 0.8352 | [0.822, 0.848] | 0.8351 | 0.8359 |
| Pearson / Kendall tau-b / MAE | 0.7711 / 0.6170 / 0.2860 | | 0.7711 / 0.6169 / 0.2860 | |
| Pairwise accuracy, pairs with human gap > 0.75 (153,238 pairs) | 0.9626 | | 0.9625 | |

The metrics are the vendored upstream `syncreward.metrics` functions; the official column is the unmodified `evaluate.py` (single process,
batch size 8, `configs/eval.yaml`; torch 2.9.1, transformers 5.1.0, FFmpeg 7 shared DLLs). Tables by generator, by human gap, best-of-N for N = 2, 3, 4, 6 and the 70-clip
controlled-change experiment (shifts, reversal, silence, noise, other audio, gain, six re-encodings) are in [LIMITATIONS.md](../LIMITATIONS.md);
the per-clip numbers are in [docs/eval/](eval/). Setup of the controlled-change experiment: 10 clips per generator with a human target of at least 1.5
(fixed seed), each scored 19 times; the false alarm threshold is taken from the drops caused by the two changes that should not matter
(-12 dB gain, x264 crf 18 + AAC 192k).

## 10. A decoding mismatch found with the benchmark clips, and its fix (0.1.1)

The ten demo clips could not show it (they all have a 23.999998413 fps stream, a container as long as the picture and a 1024-sample skip; checked);
the 1,000 benchmark clips, made by seven different generators, did. Comparing the pack with the
official loader and `evaluate.py` clip by clip showed that 0.1.0 decoded two details differently:

| | official `load_av` (torchcodec) | pack 0.1.0 | pack 0.1.1 |
| --- | --- | --- | --- |
| frame rate | frames / duration of the **video stream**, rounded to microseconds (145 frames in 6.041666... s: 23.99999867586214) | frames / duration of the **container** | frames / `round(video stream duration, 6)`; equal to torchcodec's `average_fps`, bit for bit, on all 1,000 clips |
| leading audio | the first packet states the samples to skip; some files (all 143 kling clips) say **2048** on a first frame that holds 1024 and the second frame (pts -1024) is dropped too | dropped the skip from the first frame only: 1024 samples (23 ms) too many | the skip is dropped from the whole track |

Affected in SyncReward-Bench: the frame rate of 428 clips (all davinci, nava and seedance clips, because their audio is longer than the picture) and
the audio start of 143 (kling): 571 of 1,000. With 0.1.0, the pack differed from the official `evaluate.py` clip by clip (different environment
included) by up to 0.22 (mean 0.011; 388 of 1,000 identical). The benchmark metrics barely moved (Spearman 0.7801 -> 0.7810), the
individual rewards did. The ten demo clips are unchanged (their containers and streams agree).

After the fix, `scripts/eval_suite.py bench` against the official `evaluate.py` (batch size 8; `bench_official_t29`):

| pack environment | identical | within one bfloat16 step (0.0078) | largest difference |
| --- | --- | --- | --- |
| official (torch 2.9.1, transformers 5.1.0) | 842 / 1000 | **1000 / 1000** | 0.0078 |
| ComfyUI (torch 2.14.1, transformers 5.20.1) | 726 / 1000 | 990 / 1000 | 0.0234 (one clip above 0.02) |

Through the real ComfyUI queue (clean install updated to commit `b56f7fc`, ComfyUI v0.38.0, torch 2.14.1, transformers 5.20.1; `scripts/queue_scores.py`): 31
videos (3 benchmark clips per generator + the 10 demo clips) all succeeded; compared with the official script's rewards 24 are identical, 30 of 31 are within one
bfloat16 step and the largest difference is 0.0195 (one davinci clip; the same clip agrees within one step in the official environment). The three shipped
API workflows (`score_basic`, `shift_compare`, `select_best`) ran. The *Select Best* example picked the clip whose reward was 0.008 above the others
(1.1562 against 1.1484 twice), a difference that is inside the noise described in LIMITATIONS.md.

The official script itself moves by one step with its batch size (batch 1 vs batch 8 on the 285 nava and seedance clips: 260 identical, all within
0.0078), so the remaining one-step differences are not decoding differences. Regression tests: `tests/test_media.py` now checks a video whose
audio is longer than its picture, a first-packet skip of 2048 over two frames and the microsecond rounding of the frame rate.

## Commands

```
python scripts/fetch_bench_clips.py --data-dir <dataset jsonl files> --out <dir>                                                  # section 9
python scripts/eval_suite.py bench|perturb|report ...                                                                            # section 9 (docs/eval/README.md)
python scripts/repro_timm_labels.py <folder with the pe-av-large config.json>                                                    # section 4
python tools/download_models.py --models-dir <ComfyUI>/models/syncreward
python scripts/parity_vs_official.py --upstream <SyncReward checkout> --checkpoint <syncreward-stage2> --encoder <pe-av-large> --ffmpeg-dll-dir <FFmpeg shared bin> clip.mp4 ...
python scripts/queue_scores.py --server http://127.0.0.1:8188 --comfy-input <ComfyUI>/input --official official.jsonl clip.mp4 ...
python scripts/experiment_audio_offset.py --server http://127.0.0.1:8188 --comfy-input <ComfyUI>/input --out offsets.json clip.mp4 ...
python scripts/experiment_controls.py --comfyui <ComfyUI> --checkpoint <syncreward-stage2> --encoder <pe-av-large> clip.mp4 ...
python scripts/ci_queue_check.py --comfyui <ComfyUI>
```
