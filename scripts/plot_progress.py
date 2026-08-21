from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "reports" / "training_scores.csv",
    ROOT / "reports" / "training_scores_resume.csv",
]
DEFAULT_BASELINE = ROOT / "reports" / "random_baseline.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "progression_curve.svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the training progression curve as SVG.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Training CSV files to include, in chronological/report order.",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", type=int, default=25, help="Moving-average window for train scores.")
    return parser.parse_args()


def read_training(paths: list[Path]) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    train_scores: dict[int, list[float]] = defaultdict(list)
    eval_scores: dict[int, list[float]] = defaultdict(list)

    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if not row.get("episode") or not row.get("score"):
                    continue
                episode = int(row["episode"])
                score = float(row["score"])
                if row.get("kind") == "eval":
                    eval_scores[episode].append(score)
                elif row.get("kind") == "train":
                    train_scores[episode].append(score)
    return train_scores, eval_scores


def read_baseline(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as file:
        scores = [float(row["score"]) for row in csv.DictReader(file) if row.get("score")]
    return statistics.mean(scores) if scores else None


def average_by_episode(scores: dict[int, list[float]]) -> list[tuple[int, float]]:
    return [(episode, statistics.mean(values)) for episode, values in sorted(scores.items())]


def moving_average(points: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    if window <= 1:
        return points
    smoothed: list[tuple[int, float]] = []
    values: list[float] = []
    for episode, score in points:
        values.append(score)
        current = values[-window:]
        smoothed.append((episode, statistics.mean(current)))
    return smoothed


def polyline(points: list[tuple[int, float]], xscale, yscale) -> str:
    return " ".join(f"{xscale(x):.1f},{yscale(y):.1f}" for x, y in points)


def write_svg(
    output: Path,
    train: list[tuple[int, float]],
    evals: list[tuple[int, float]],
    baseline: float | None,
) -> None:
    width = 1100
    height = 620
    left = 80
    right = 30
    top = 60
    bottom = 80
    chart_w = width - left - right
    chart_h = height - top - bottom

    all_points = train + evals
    if baseline is not None:
        all_points.append((train[-1][0] if train else 1, baseline))
    if not all_points:
        raise ValueError("No score data found.")

    min_episode = min(x for x, _ in all_points)
    max_episode = max(x for x, _ in all_points)
    max_score = max(y for _, y in all_points)
    y_max = max(20.0, math.ceil(max_score / 5.0) * 5.0)

    def xscale(x: int) -> float:
        if max_episode == min_episode:
            return left + chart_w / 2
        return left + (x - min_episode) / (max_episode - min_episode) * chart_w

    def yscale(y: float) -> float:
        return top + chart_h - (y / y_max) * chart_h

    eval_best: list[tuple[int, float]] = []
    best = float("-inf")
    for episode, score in evals:
        best = max(best, score)
        eval_best.append((episode, best))

    grid = []
    for i in range(0, int(y_max) + 1, 5):
        y = yscale(i)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" '
            'font-size="12" fill="#4b5563">'
            f"{i}</text>"
        )

    x_ticks = []
    span = max_episode - min_episode
    tick_step = 100 if span <= 600 else 200
    start_tick = (min_episode // tick_step) * tick_step
    for episode in range(start_tick, max_episode + 1, tick_step):
        if episode < min_episode:
            continue
        x = xscale(episode)
        x_ticks.append(
            f'<line x1="{x:.1f}" y1="{top + chart_h}" x2="{x:.1f}" '
            f'y2="{top + chart_h + 6}" stroke="#4b5563" stroke-width="1"/>'
        )
        x_ticks.append(
            f'<text x="{x:.1f}" y="{top + chart_h + 24}" text-anchor="middle" '
            'font-size="12" fill="#4b5563">'
            f"{episode}</text>"
        )

    train_path = polyline(train, xscale, yscale)
    eval_path = polyline(evals, xscale, yscale)
    best_path = polyline(eval_best, xscale, yscale)
    baseline_y = yscale(baseline) if baseline is not None else None

    eval_circles = "\n".join(
        f'<circle cx="{xscale(x):.1f}" cy="{yscale(y):.1f}" r="3.5" fill="#dc2626"/>'
        for x, y in evals
    )
    baseline_line = ""
    baseline_label = ""
    if baseline_y is not None and baseline is not None:
        baseline_line = (
            f'<line x1="{left}" y1="{baseline_y:.1f}" x2="{width - right}" '
            f'y2="{baseline_y:.1f}" stroke="#6b7280" stroke-width="2" '
            'stroke-dasharray="7 5"/>'
        )
        baseline_label = (
            f'<text x="{width - right}" y="{baseline_y - 8:.1f}" text-anchor="end" '
            'font-size="13" fill="#374151">'
            f"Baseline aleatoire: {baseline:.3f}</text>"
        )

    final_eval = evals[-1][1] if evals else 0.0
    best_eval = max((score for _, score in evals), default=0.0)
    title = "Progression de l'agent Q-learning"
    subtitle = (
        f"Train moyenne mobile, eval checkpoints, meilleur eval {best_eval:.3f}, "
        f"dernier eval {final_eval:.3f}"
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="28" font-size="24" font-weight="700" fill="#111827">{title}</text>
  <text x="{left}" y="50" font-size="13" fill="#4b5563">{subtitle}</text>
  <rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="#f9fafb" stroke="#d1d5db"/>
  {"".join(grid)}
  {"".join(x_ticks)}
  {baseline_line}
  {baseline_label}
  <polyline points="{train_path}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
  <polyline points="{eval_path}" fill="none" stroke="#dc2626" stroke-width="2.5"/>
  <polyline points="{best_path}" fill="none" stroke="#16a34a" stroke-width="2" stroke-dasharray="5 4"/>
  {eval_circles}
  <text x="{left + chart_w / 2:.1f}" y="{height - 22}" text-anchor="middle" font-size="14" fill="#374151">Episode</text>
  <text x="24" y="{top + chart_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 24 {top + chart_h / 2:.1f})" font-size="14" fill="#374151">Score</text>
  <rect x="{width - 315}" y="72" width="270" height="86" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{width - 298}" y1="94" x2="{width - 258}" y2="94" stroke="#2563eb" stroke-width="3"/>
  <text x="{width - 248}" y="99" font-size="13" fill="#111827">Train, moyenne mobile</text>
  <line x1="{width - 298}" y1="119" x2="{width - 258}" y2="119" stroke="#dc2626" stroke-width="3"/>
  <text x="{width - 248}" y="124" font-size="13" fill="#111827">Evaluation moyenne</text>
  <line x1="{width - 298}" y1="144" x2="{width - 258}" y2="144" stroke="#16a34a" stroke-width="2" stroke-dasharray="5 4"/>
  <text x="{width - 248}" y="149" font-size="13" fill="#111827">Meilleur eval conserve</text>
</svg>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_scores, eval_scores = read_training(args.input)
    train = moving_average(average_by_episode(train_scores), args.window)
    evals = average_by_episode(eval_scores)
    baseline = read_baseline(args.baseline)
    write_svg(args.output, train, evals, baseline)
    print(f"wrote {args.output}")
    print(f"train_points={len(train)} eval_points={len(evals)} baseline={baseline:.3f}" if baseline else "")


if __name__ == "__main__":
    main()
