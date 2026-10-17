"""scripts/eval_suite.py: the statistics and the report on synthetic numbers (no model, no video, no GPU)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_suite  # noqa: E402


def test_auc_counts_ties_as_half():
    assert eval_suite.auc(np.array([2.0, 3.0]), np.array([1.0])) == 1.0
    assert eval_suite.auc(np.array([1.0]), np.array([1.0])) == 0.5
    assert eval_suite.auc(np.array([1.0, 3.0]), np.array([2.0])) == 0.5


def test_bootstrap_interval_contains_the_mean():
    values = np.linspace(0, 1, 50)
    lo, hi = eval_suite.bootstrap_mean(values)
    assert lo < values.mean() < hi


def _write(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_report_on_synthetic_perturbations(tmp_path, capsys):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        base = 1.5 + 0.1 * rng.standard_normal()
        model = "gen_a" if i < 6 else "gen_b"
        effects = {"original": 0.0, "shift_+1000ms": 0.05, "silence": 0.8, "reversed": 0.0, "gain_-12dB": 0.0, "enc_crf18_aac192": 0.0}
        for variant, drop in effects.items():
            noise = 0.01 * rng.standard_normal() if variant != "original" else 0.0
            rows.append({"id": f"c{i}", "model": model, "human_target": 1.8, "variant": variant, "reward": base - drop + noise})
    rows.append({"id": "c0", "model": "gen_a", "variant": "enc_fps25", "error": "RuntimeError: boom"})
    perturb = tmp_path / "perturb.jsonl"
    _write(perturb, rows)
    md = tmp_path / "report.md"
    (tmp_path / "bench_syncreward_bench.jsonl").write_text("", encoding="utf-8")
    assert eval_suite.main(["report", "--perturb", str(perturb), "--data-dir", str(tmp_path), "--md", str(md)]) == 0
    text = md.read_text(encoding="utf-8")
    assert "12 clips" in text and "1 variants failed" in text
    silence = next(line for line in text.splitlines() if line.startswith("| silence |"))
    cells = [c.strip() for c in silence.strip("|").split("|")]
    assert float(cells[3].split()[0]) > 0.7  # mean drop of about 0.8
    assert float(cells[5]) == 1.0  # the original is always above silence
    assert float(cells[6]) == 1.0  # and the silent copy is detected at a 5 % false alarm rate
    reversed_line = next(line for line in text.splitlines() if line.startswith("| reversed |"))
    assert float([c.strip() for c in reversed_line.strip("|").split("|")][6]) < 0.5  # an undetectable change stays undetected


def test_report_metrics_on_synthetic_benchmark(tmp_path):
    rows, preds = [], []
    rng = np.random.default_rng(1)
    for i in range(40):
        target = float(rng.integers(0, 5)) / 2
        rows.append({"id": f"curated1k_{i:04d}", "video_path": f"generated/g/{i}.mp4", "uid": f"u{i}", "model": "g" if i < 20 else "h", "target": target})
        preds.append({"id": f"curated1k_{i:04d}", "reward": target + 0.1 * float(rng.standard_normal())})
    _write(tmp_path / "bench_syncreward_bench.jsonl", rows)
    _write(tmp_path / "bench.jsonl", preds + [{"id": "curated1k_9999", "error": "x"}])
    md = tmp_path / "bench.md"
    assert eval_suite.main(["report", "--bench", str(tmp_path / "bench.jsonl"), "--data-dir", str(tmp_path), "--md", str(md)]) == 0
    text = md.read_text(encoding="utf-8")
    assert "40 of 40 clips scored, 1 failed" in text
    spearman = next(line for line in text.splitlines() if line.startswith("| spearman |"))
    assert float(spearman.split("|")[2].split()[0]) > 0.9
