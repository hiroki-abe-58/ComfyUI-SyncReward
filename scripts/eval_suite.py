"""Evaluation suite for the SyncReward pack: human-rated benchmark, and how the reward reacts to controlled audio changes.

All scoring runs in-process through the pack's own decode and scoring code (`syncreward_comfy.media` / `.runtime`, the code behind the
nodes), with the real weights. Needs the clips of SyncReward-Bench (scripts/fetch_bench_clips.py) and, for the encoding variants, an
ffmpeg executable with libx264 and aac.

    python scripts/eval_suite.py bench   --data-dir <dir> --video-root <dir> --checkpoint <folder> --encoder <folder> --out bench.jsonl
    python scripts/eval_suite.py perturb --data-dir <dir> --video-root <dir> --checkpoint <folder> --encoder <folder> --ffmpeg <ffmpeg> --out perturb.jsonl
    python scripts/eval_suite.py report  --bench bench.jsonl --data-dir <dir> --perturb perturb.jsonl [--md report.md]

`bench` scores every benchmark clip (resumable) and, in `report`, computes the paper's metrics with the vendored upstream
`syncreward.metrics`. `perturb` takes clips that humans rated as well synchronized (target >= --min-target, equal numbers per
generator) and scores, per clip: the original; the audio delayed or advanced by 0.2 / 0.5 / 1.0 s; reversed; silent; white noise of
the same loudness; the audio of another clip of the same generator and of a clip of another generator; the original 12 dB quieter;
and six re-encodings of the same clip (one of high quality, which should change nothing, and five that lower the picture or sound
quality or change the frame rate). Nothing is written but the numbers; re-encoded files are temporary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SHIFTS_MS = [-1000, -500, -200, 200, 500, 1000]
ENCODINGS = {
    "enc_crf18_aac192": ["-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-b:a", "192k"],
    "enc_crf30_aac64": ["-c:v", "libx264", "-crf", "30", "-c:a", "aac", "-b:a", "64k"],
    "enc_half_res": ["-vf", "scale=trunc(iw/4)*2:trunc(ih/4)*2", "-c:v", "libx264", "-crf", "23", "-c:a", "aac", "-b:a", "128k"],
    "enc_fps25": ["-r", "25", "-c:v", "libx264", "-crf", "23", "-c:a", "aac", "-b:a", "128k"],
    "enc_audio16k_mono": ["-c:v", "libx264", "-crf", "23", "-c:a", "aac", "-b:a", "48k", "-ar", "16000", "-ac", "1"],
    "enc_mpeg4_mp3": ["-c:v", "mpeg4", "-q:v", "4", "-c:a", "libmp3lame", "-b:a", "128k"],
}
EQUIVALENT = ["gain_-12dB", "enc_crf18_aac192"]  # the changes that should not matter (the other encodings degrade the picture or the sound)
DEGRADING = [name for name in ENCODINGS if name not in EQUIVALENT]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_pack(args):
    sys.path.insert(0, str(ROOT))
    if getattr(args, "comfyui", None):
        sys.path.insert(1, str(args.comfyui))
    from syncreward_comfy import media, runtime

    loaded = runtime.acquire(runtime.make_handle(str(args.checkpoint), str(args.encoder), "auto", True))
    return media, runtime, loaded


def mono(media, loaded, own, transform=None):
    if transform is not None:
        own = {"waveform": transform(own["waveform"]), "sample_rate": own["sample_rate"]}
    return media.mono_audio(own, loaded.sample_rate)


def cmd_bench(args) -> int:
    media, runtime, loaded = load_pack(args)
    rows = read_jsonl(args.data_dir / "bench_syncreward_bench.jsonl")
    done = {r["id"] for r in read_jsonl(args.out)} if args.out.exists() else set()
    started = time.time()
    with args.out.open("a", encoding="utf-8") as sink:
        for n, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            path = args.video_root / row["video_path"]
            if not path.exists():  # not downloaded (yet): leave it for a later, resumed run
                continue
            try:
                frames, fps, own = media.decode_file(str(path))
                report = runtime.score(loaded, frames, fps, media.mono_audio(own, loaded.sample_rate))
                out = {"id": row["id"], "video_path": row["video_path"], "reward": report["reward"], "video_seconds": report["video_seconds"],
                       "audio_seconds": report["audio_seconds"], "scoring_seconds": report["scoring_seconds"],
                       "shorter_than_window": report["shorter_than_window"]}
            except Exception as exc:  # noqa: BLE001 - record and go on, the report counts the failures
                out = {"id": row["id"], "video_path": row["video_path"], "error": f"{type(exc).__name__}: {exc}"}
            sink.write(json.dumps(out) + "\n")
            sink.flush()
            if n % 50 == 0:
                print(f"{n}/{len(rows)} clips, {time.time() - started:.0f} s", flush=True)
    return 0


def pick_clips(args):
    rows = read_jsonl(args.data_dir / "bench_syncreward_bench.jsonl")
    rng = np.random.default_rng(args.seed)
    chosen = []
    for model in sorted({r["model"] for r in rows}):
        pool = [r for r in rows if r["model"] == model and r["target"] >= args.min_target and (args.video_root / r["video_path"]).exists()]
        pool.sort(key=lambda r: r["id"])
        pick = rng.permutation(len(pool))[: args.per_generator]
        chosen += [pool[i] for i in sorted(pick)]
    return chosen


def encode(ffmpeg: str, source: Path, target: Path, options: list[str]) -> None:
    command = [ffmpeg, "-v", "error", "-y", "-i", str(source), *options, str(target)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-300:])


def cmd_perturb(args) -> int:
    media, runtime, loaded = load_pack(args)
    clips = pick_clips(args)
    print(f"{len(clips)} clips: {sorted({c['model'] for c in clips})}", flush=True)
    done = set()
    if args.out.exists():
        done = {(r["id"], r["variant"]) for r in read_jsonl(args.out)}
    audio_of = {}  # mono audio of every selected clip (frames are decoded one clip at a time to keep memory flat)
    for c in clips:
        _, _, own_audio = media.decode_file(str(args.video_root / c["video_path"]))
        audio_of[c["id"]] = media.mono_audio(own_audio, loaded.sample_rate)
    work = Path(tempfile.mkdtemp(prefix="syncreward_eval_"))
    started = time.time()
    try:
        with args.out.open("a", encoding="utf-8") as sink:
            for i, clip in enumerate(clips):
                frames, fps, own = media.decode_file(str(args.video_root / clip["video_path"]))
                rate = int(own["sample_rate"])
                base = mono(media, loaded, own)
                rms = float(np.sqrt((base ** 2).mean())) or 1e-4
                rng = np.random.default_rng([args.seed, i])
                same_gen = [c for c in clips if c["model"] == clip["model"] and c["id"] != clip["id"]]
                other_gen = [c for c in clips if c["model"] != clip["model"]]
                donors = {"other_clip_same_generator": same_gen[i % len(same_gen)], "other_clip_other_generator": other_gen[(i * 7) % len(other_gen)]}
                variants = {"original": (frames, fps, base)}
                for ms in SHIFTS_MS:
                    samples = int(round(ms / 1000.0 * rate))
                    variants[f"shift_{ms:+d}ms"] = (frames, fps, mono(media, loaded, own, lambda w, s=samples: media.shift_waveform(w, s)))
                variants["reversed"] = (frames, fps, base[::-1].copy())
                variants["silence"] = (frames, fps, np.zeros_like(base))
                variants["white_noise_same_rms"] = (frames, fps, (rng.standard_normal(base.shape) * rms).astype(np.float32))
                for label, donor in donors.items():
                    donor_audio = audio_of[donor["id"]]
                    variants[label] = (frames, fps, np.resize(donor_audio, base.shape).astype(np.float32))
                variants["gain_-12dB"] = (frames, fps, (base * 10 ** (-12 / 20)).astype(np.float32))
                for label in variants:
                    if (clip["id"], label) in done:
                        continue
                    f, r, a = variants[label]
                    reward = runtime.score(loaded, f, r, a)["reward"]
                    sink.write(json.dumps({"id": clip["id"], "model": clip["model"], "human_target": clip["target"], "variant": label, "reward": reward}) + "\n")
                for label, options in ENCODINGS.items():
                    if (clip["id"], label) in done:
                        continue
                    target = work / f"{clip['id']}_{label}.{'avi' if 'mpeg4' in options else 'mp4'}"
                    try:
                        encode(args.ffmpeg, args.video_root / clip["video_path"], target, options)
                        f, r, own2 = media.decode_file(str(target))
                        reward = runtime.score(loaded, f, r, media.mono_audio(own2, loaded.sample_rate))["reward"]
                        sink.write(json.dumps({"id": clip["id"], "model": clip["model"], "human_target": clip["target"], "variant": label, "reward": reward}) + "\n")
                    except Exception as exc:  # noqa: BLE001
                        sink.write(json.dumps({"id": clip["id"], "model": clip["model"], "variant": label, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                    finally:
                        target.unlink(missing_ok=True)
                sink.flush()
                print(f"clip {i + 1}/{len(clips)} ({clip['id']}) done, {time.time() - started:.0f} s", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(positive > negative), ties count half (Mann-Whitney)."""
    diff = positive[:, None] - negative[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def bootstrap_mean(values: np.ndarray, seed: int = 0, n: int = 2000):
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cmd_report(args) -> int:
    lines = []
    out = lambda s="": (print(s), lines.append(s))  # noqa: E731
    if args.bench:
        sys.path.insert(0, str(ROOT))
        from syncreward_comfy.vendor.syncreward import metrics

        manifest = {r["id"]: r for r in read_jsonl(args.data_dir / "bench_syncreward_bench.jsonl")}
        preds = read_jsonl(args.bench)
        bad = [p for p in preds if "reward" not in p]
        good = [p for p in preds if "reward" in p]
        target = [manifest[p["id"]]["target"] for p in good]
        reward = [p["reward"] for p in good]
        m = metrics.compute_metrics(target, reward)
        m["ci95"] = metrics.bootstrap_ci(target, reward, 1000)
        out(f"## SyncReward-Bench: {len(good)} of {len(manifest)} clips scored, {len(bad)} failed")
        out("| metric | this pack | model card (Stage 2) |")
        out("| --- | --- | --- |")
        card = {"spearman": 0.7823, "top1_n4": 0.7646, "pair_acc": 0.8359}
        for k in ("spearman", "top1_n4", "pair_acc", "pearson", "kendall", "pair_acc_gap0.75", "mae"):
            ci = m["ci95"].get(k)
            out(f"| {k} | {m[k]:.4f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "") + f" | {card.get(k, '')} |")
        out(f"\nchance level for top-1 (N=4): {m['top1_n4_chance']:.4f}; pairs: {m['pair_n']}, pairs with target gap > 0.75: {m['pair_n_gap0.75']}")
        out("\n| generator | clips | Spearman | pair acc | mean target | mean reward |")
        out("| --- | --- | --- | --- | --- | --- |")
        for model in sorted({manifest[p["id"]]["model"] for p in good}):
            sel = [p for p in good if manifest[p["id"]]["model"] == model]
            t, r = [manifest[p["id"]]["target"] for p in sel], [p["reward"] for p in sel]
            out(f"| {model} | {len(sel)} | {metrics.spearman(r, t):.3f} | {metrics.pairwise_accuracy(t, r)[0]:.3f} | {np.mean(t):.2f} | {np.mean(r):.2f} |")
        t, r = np.array(target), np.array(reward)
        rows_i, cols_i = np.triu_indices(len(t), k=1)
        gap, diff = np.abs(t[rows_i] - t[cols_i]), np.sign(t[rows_i] - t[cols_i]) * np.sign(r[rows_i] - r[cols_i])
        out("\nPairwise accuracy by size of the human gap (a tie of the reward counts half):\n")
        out("| human target gap | pairs | reward orders them like the humans |")
        out("| --- | --- | --- |")
        for lo_g, hi_g in ((0.0, 0.34), (0.34, 0.67), (0.67, 1.01), (1.01, 2.01)):
            pick = (gap > max(lo_g, 1e-9)) & (gap <= hi_g)
            if pick.any():
                out(f"| {lo_g:.2f} < gap <= {hi_g:.2f} | {int(pick.sum())} | {((diff[pick] > 0).sum() + 0.5 * (diff[pick] == 0).sum()) / pick.sum():.3f} |")
        out("\nBest-of-N selection among random groups of benchmark clips (as in *Select Best*; a group counts as right when the top-scored clip is one of the "
            "human-best ones, ties of the reward share the credit; 1000 random groupings):\n")
        out("| N | picks a human-best clip | chance |")
        out("| --- | --- | --- |")
        for n_group in (2, 3, 4, 6):
            acc, chance = metrics.top1_accuracy(t, r, group_size=n_group, draws=1000)
            out(f"| {n_group} | {acc:.3f} | {chance:.3f} |")
        high, low = r[t >= 1.5], r[t <= 0.5]
        out(f"\nreward of clips humans rated >= 1.5 (n={len(high)}): mean {high.mean():.3f}; rated <= 0.5 (n={len(low)}): mean {low.mean():.3f}; "
            f"P(high-rated clip scores above low-rated clip) = {auc(high, low):.3f}")
        out("")
    if args.perturb:
        rows = [r for r in read_jsonl(args.perturb) if "reward" in r]
        failed = [r for r in read_jsonl(args.perturb) if "reward" not in r]
        by = {}
        for r in rows:
            by.setdefault(r["variant"], {})[r["id"]] = r["reward"]
        ids = sorted(by["original"])
        base = np.array([by["original"][i] for i in ids])
        out(f"## Controlled changes: {len(ids)} clips rated >= {args.min_target} by humans; {len(failed)} variants failed")
        out(f"original reward: mean {base.mean():.3f}, SD {base.std():.3f}, min {base.min():.3f}, max {base.max():.3f}\n")
        equivalent = np.array([by["original"][i] - by[v][i] for v in EQUIVALENT if v in by for i in ids if i in by[v]])
        thr = {fa: float(np.quantile(equivalent, 1 - fa)) for fa in (0.05, 0.01)}
        out(f"harmless changes (gain -12 dB, high-quality re-encode crf 18 / AAC 192k; {len(equivalent)} pairs): reward drop original - variant has mean "
            f"{equivalent.mean():+.4f}, SD {equivalent.std():.4f}; 95th percentile {thr[0.05]:.4f}, 99th {thr[0.01]:.4f}\n")
        out("| change | n | mean reward | mean drop vs original [95% CI] | original higher | P(original > variant) | detected at 5% false alarm | at 1% |")
        out("| --- | --- | --- | --- | --- | --- | --- | --- |")
        order = ["original", *[f"shift_{ms:+d}ms" for ms in sorted(SHIFTS_MS)], "reversed", "silence", "white_noise_same_rms",
                 "other_clip_same_generator", "other_clip_other_generator", *EQUIVALENT, *DEGRADING]
        for v in order:
            if v not in by or v == "original":
                continue
            common = [i for i in ids if i in by[v]]
            b = np.array([by["original"][i] for i in common])
            x = np.array([by[v][i] for i in common])
            drop = b - x
            lo, hi = bootstrap_mean(drop)
            out(f"| {v} | {len(common)} | {x.mean():.3f} | {drop.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {(drop > 0).mean():.2f} | {auc(b, x):.3f} | "
                f"{(drop > thr[0.05]).mean():.2f} | {(drop > thr[0.01]).mean():.2f} |")
        out("\n'original higher' = share of clips whose original reward is above the variant's; P(original > variant) is the same over all "
            "pairs of clips (a tie counts half). A selection between the original and a changed copy is wrong when the changed copy scores at least as high.")
        out("\n| generator | clips | mean original | mean drop at 1.0 s (either direction) | mean drop for silence |")
        out("| --- | --- | --- | --- | --- |")
        model_of = {r["id"]: r["model"] for r in rows}
        for model in sorted(set(model_of.values())):
            sel = [i for i in ids if model_of[i] == model]
            o = np.array([by["original"][i] for i in sel])
            s1 = [np.mean([by["original"][i] - by[f"shift_{s:+d}ms"][i] for s in (-1000, 1000) if i in by.get(f"shift_{s:+d}ms", {})]) for i in sel]
            sil = [by["original"][i] - by["silence"][i] for i in sel if i in by.get("silence", {})]
            out(f"| {model} | {len(sel)} | {o.mean():.3f} | {np.mean(s1):+.3f} | {np.mean(sil):+.3f} |")
    if args.md:
        args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("bench", "perturb"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--video-root", type=Path, required=True)
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--encoder", type=Path, required=True)
        p.add_argument("--comfyui", type=Path, help="ComfyUI checkout (supplies comfy.audio.resample)")
        p.add_argument("--out", type=Path, required=True)
        if name == "perturb":
            p.add_argument("--ffmpeg", default="ffmpeg")
            p.add_argument("--min-target", type=float, default=1.5)
            p.add_argument("--per-generator", type=int, default=10)
            p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("--bench", type=Path)
    p.add_argument("--perturb", type=Path)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--min-target", type=float, default=1.5)
    p.add_argument("--md", type=Path)
    args = parser.parse_args(argv)
    return {"bench": cmd_bench, "perturb": cmd_perturb, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
