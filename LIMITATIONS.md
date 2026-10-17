# What the SyncReward score can and cannot tell you (measured)

Read this before you use **SyncReward Select Best** to choose between videos, or the score as a pass/fail test. All numbers below come
from this pack running the real Stage-2 weights on an RTX 5090 (Windows 11, ComfyUI environment with torch 2.14.1 and transformers 5.20.1),
on clips of the official SyncReward-Bench. The commands are in [docs/VERIFICATION.md](docs/VERIFICATION.md) (`scripts/eval_suite.py`); the raw
per-clip numbers are in [docs/eval/](docs/eval/). The clips themselves are not included (they belong to the dataset and its generators' terms).

## What the model is for

SyncReward was trained to predict how well humans rate the **audio-visual synchronization of generated clips** (text-to-audio-video
generators, 8 of them, 5 s clips). Its model card says it scores only the temporal alignment of on-screen sound sources inside a 4.75 s
window and does not judge audio quality or semantic relevance. **Outside that range this pack has not measured anything**: real camera
footage, clips much shorter or longer than 5 s, other generators, other languages or sound types than the benchmark contains.

## 1. Agreement with human raters: the benchmark is reproduced

1,000 clips of SyncReward-Bench (7 generators, target = mean of up to 3 human ratings), all scored by this pack:

| metric | this pack (95 % bootstrap interval) | the official `evaluate.py`, run here | model card, Stage 2 |
| --- | --- | --- | --- |
| Spearman correlation | 0.781 [0.756, 0.805] | 0.7810 | 0.7823 |
| Top-1 (best of N = 4) | 0.764 [0.713, 0.821] | 0.7641 | 0.7646 |
| Pairwise accuracy | 0.835 [0.822, 0.848] | 0.8351 | 0.8359 |

The three numbers are inside each other's intervals. Even the unmodified official script, run on this machine, gives 0.7810 / 0.7641 / 0.8351
and not the model card's values (different operating system, library versions and bfloat16 rounding), and the pack stays within 0.0002 of it.
Clip by clip, in the official environment (torch 2.9.1, transformers 5.1.0) the rewards of the pack and of the official script differ by at most
one bfloat16 step (0.008) on all 1,000 clips, and are identical on 842 of them; in a current ComfyUI environment (torch 2.14.1, transformers
5.20.1) 990 of 1,000 are within one step (726 identical) and the largest difference is 0.023 (docs/VERIFICATION.md, section 10).

How reliable is a single comparison? Pairs of benchmark clips grouped by how far apart the humans rated them (reward ties count half):

| human target gap | pairs | reward orders them like the humans |
| --- | --- | --- |
| up to 0.33 (at most a third of a rating step) | 147,650 | **0.694** |
| 0.34 to 0.67 | 120,860 | 0.846 |
| 0.68 to 1.01 | 81,398 | 0.941 |
| above 1.01 | 71,840 | 0.987 |

Best-of-N over random groups of benchmark clips (a group is right when the top-scored clip is one of the human-best ones):

| N | picks a human-best clip | chance |
| --- | --- | --- |
| 2 | 0.836 | 0.500 |
| 3 | 0.800 | 0.404 |
| 4 | 0.764 | 0.332 |
| 6 | 0.714 | 0.248 |

By generator (Spearman / pairwise accuracy): davinci 0.67 / 0.78, kling 0.76 / 0.84, ltx2.3 0.79 / 0.85, mova 0.68 / 0.80, nava 0.73 / 0.82,
ovi 0.76 / 0.82, **seedance 0.60 / 0.76**.

**What this does not say.** The groups above are random clips of different prompts and generators, so many differences are large. The
case *Select Best* is built for, several candidates of the **same prompt from the same generator**, was **not measured** (the benchmark has
no such groups; the few prompts that appear twice are rendered by different generators). Such candidates tend to be closer together,
and the table by human gap suggests that close pairs are ordered correctly only about 7 times in 10.

## 2. What the reward reacts to: controlled changes of well-synchronized clips

70 clips that the humans rated 1.5 or higher (10 per generator, chosen with a fixed seed), each scored as is and with one change. The
picture is never touched, except in the re-encodings. "Detected" means the reward fell by more than the 95th (or 99th) percentile of
the drops caused by changes that should not matter (-12 dB gain and a high-quality re-encode: 140 pairs, drop SD 0.052, 95th percentile 0.110, 99th 0.183),
so the false alarm rate is 5 % (1 %) by construction. Original reward: mean 1.54, SD 0.30.

| change to the clip | mean drop in reward [95 % CI] | the original scored higher in | detected (5 % / 1 % false alarms) |
| --- | --- | --- | --- |
| audio delayed 0.2 s | -0.000 [-0.005, +0.005] | 36 % | 0 % / 0 % |
| audio advanced 0.2 s | +0.001 [-0.004, +0.006] | 46 % | 0 % / 0 % |
| audio delayed 0.5 s | +0.005 [-0.001, +0.010] | 49 % | 0 % / 0 % |
| audio advanced 0.5 s | +0.006 [+0.001, +0.011] | 50 % | 0 % / 0 % |
| audio delayed 1.0 s | +0.014 [+0.004, +0.025] | 57 % | 1 % / 1 % |
| audio advanced 1.0 s | +0.012 [+0.005, +0.020] | 53 % | 3 % / 0 % |
| audio played backwards | +0.063 [+0.051, +0.076] | 83 % | 19 % / 0 % |
| silence | +0.256 [+0.183, +0.339] | 81 % | 57 % / 46 % |
| white noise of the same loudness | +0.255 [+0.209, +0.304] | 93 % | 67 % / 60 % |
| audio of another clip of the same generator | +0.124 [+0.069, +0.191] | 70 % | 31 % / 26 % |
| audio of a clip of another generator | +0.134 [+0.044, +0.237] | 59 % | 37 % / 34 % |
| -12 dB gain (should not matter) | -0.002 [-0.016, +0.012] | 43 % | 4 % / 1 % |
| re-encode x264 crf 18, AAC 192k (should not matter) | +0.019 [+0.010, +0.029] | 56 % | 6 % / 1 % |
| re-encode mpeg4 + MP3 | +0.020 [-0.001, +0.040] | 50 % | 14 % / 6 % |
| re-encode at 25 fps | +0.070 [+0.052, +0.090] | 83 % | 20 % / 13 % |
| re-encode at half resolution | +0.143 [+0.117, +0.173] | 97 % | 43 % / 29 % |
| re-encode x264 crf 30, AAC 64k | +0.206 [+0.177, +0.241] | 100 % | 71 % / 54 % |
| audio resampled to 16 kHz mono (AAC 48k) | +0.193 [+0.166, +0.221] | 99 % | 69 % / 56 % |

What follows from these numbers:

- **The reward does not measure timing offsets.** Moving the sound by up to one second in either direction changes the reward by
  at most 0.014 on average and is detected hardly more often than the false alarm rate. Do not use the score to check lip-sync, to find
  an audio delay, or to choose between two versions of the same video that differ by a small offset. The benchmark agreement in section 1 is
  therefore not evidence that the score tracks timing.
- **Missing or wrong sound is only partly caught.** A silent copy fell clearly in 57 % of the clips (at 5 % false alarms), a noisy one in
  67 %, and a copy with another clip's sound in 31 to 37 %. In about one clip in five the silent version still scored as high as the
  original (19 %; for noise 7 %, for time reversal 17 %). For the clips of one generator (ovi) silence made no difference at all
  (mean drop -0.002), for kling and seedance it dropped the reward by 0.54 and 0.59.
- **Picture and sound quality move the score nearly as much as missing sound does.** A heavy re-encode (crf 30, AAC 64k), half resolution or
  16 kHz mono audio lowers the reward by 0.14 to 0.21, against 0.26 for removing the sound (56 to 80 % of it). If your candidates differ in resolution, bitrate or
  frame rate, *Select Best* is partly choosing by that. (Why the model reacts to these is not known here; the controls were not designed
  to find out.)
- **Small differences in reward are noise.** Changes that should not matter move the reward by an SD of 0.052 (drops up to 0.11 at the
  95th percentile). Do not read a difference smaller than about 0.1 between two candidates as better or worse. bfloat16 and library
  versions add steps of 0.008 on top.

Per generator (mean drop in reward; 10 clips each): audio moved by 1.0 s (either direction): davinci +0.015, kling +0.026, ltx2.3 +0.014, mova +0.008, nava +0.005,
ovi 0.000, seedance +0.022; silence: davinci +0.120, kling +0.544, ltx2.3 +0.299, mova +0.161, nava +0.079, ovi -0.002, seedance +0.589.
## 3. Consequences for *Select Best*

- It orders candidates well when humans see a clear difference (human gap above 0.67: 94 to 99 % of pairs) and only about 7 pairs in 10 when
  the difference is small (gap up to 0.33). Treat the choice as a **suggestion** and look at the candidates; keep the individual rewards (the node outputs them) and
  ignore selections whose rewards differ by less than about 0.1.
- It will not reliably reject a video with no sound, with noise instead of sound, with someone else's sound, or with a sound that is
  late by up to a second. Check those separately (for example, compute loudness first).
- Compare candidates that were encoded the same way (same resolution, frame rate, bitrate and audio settings).
- Ties go to the lower number, a clip shorter than 4.75 s is padded, and only a centered 4.75 s window is looked at: events outside it
  do not count.

## 4. What was not measured

Agreement with human raters on candidates of the same prompt and generator; any clip that is not a generated 5 s clip of the benchmark's
kind (real recordings, other lengths, other frame rates than 24 fps, other languages); other GPUs or CPUs; the sensitivity to small
*partial* audio errors (a missing single event); timing sensitivity of clips whose humans rated them low; and whether a different
encoder, window or threshold would do better. The perturbation set is 70 clips (10 per generator), one run each; the reward is
deterministic for a given environment, so repeating it would only show bfloat16 noise.
