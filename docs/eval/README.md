# Raw numbers behind LIMITATIONS.md

Produced by `scripts/eval_suite.py` with this pack's code (version 0.1.1) and the real Stage-2 weights (RTX 5090, Windows 11, ComfyUI environment:
torch 2.14.1, transformers 5.20.1, bfloat16 autocast). **No videos or audio are included**: the clips belong to the
[SyncReward-Data](https://huggingface.co/datasets/Guan123/SyncReward-Data) dataset (annotations Apache-2.0; the videos come from third-party
generators whose terms also apply). They can be fetched one by one with `scripts/fetch_bench_clips.py`.

| File | Contents |
| --- | --- |
| `bench_predictions.jsonl` | One row per SyncReward-Bench clip (`id` and `video_path` as in `bench/syncreward_bench.jsonl` of the dataset, revision `a7904b6d5901`): the reward of this pack, clip and audio length, scoring time |
| `perturb_rewards.jsonl` | One row per clip and change (70 clips x 19): reward of the original and of each controlled change; `human_target` is the benchmark's mean human rating of that clip |
| `report.md` | The tables that `eval_suite.py report` prints from the two files |

```
python scripts/fetch_bench_clips.py --data-dir <dir with the two dataset jsonl files> --out <dir>
python scripts/eval_suite.py bench   --data-dir <dir> --video-root <dir> --checkpoint <syncreward-stage2> --encoder <pe-av-large> --out bench.jsonl
python scripts/eval_suite.py perturb --data-dir <dir> --video-root <dir> --checkpoint <syncreward-stage2> --encoder <pe-av-large> --ffmpeg <ffmpeg> --out perturb.jsonl
python scripts/eval_suite.py report  --bench bench.jsonl --perturb perturb.jsonl --data-dir <dir>
```

Scoring one 5 s clip took about 1.3 to 2 s. Rewards are deterministic for a given environment; another torch / transformers
version can move single values by one bfloat16 step (0.008).
