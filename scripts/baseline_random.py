from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eos_env import ACTIONS, EclipseEnv


DEFAULT_OUTPUT = ROOT / "reports" / "random_baseline.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the random-agent baseline.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    actions = list(ACTIONS)
    rows: list[dict[str, object]] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with EclipseEnv(headless=True, step_seconds=args.step_seconds) as env:
        for episode in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            steps = 0
            total_reward = 0.0

            while not done and steps < args.max_steps:
                action = rng.choice(actions)
                obs, reward, done, _info = env.step(action)
                total_reward += reward
                steps += 1

            score = env.get_score()
            row = {
                "episode": episode,
                "steps": steps,
                "done": done,
                "score": round(score, 6),
                "total_reward": round(total_reward, 6),
                **asdict(obs),
            }
            rows.append(row)
            print(
                f"episode={episode:02d} score={score:.3f} "
                f"reward={total_reward:.3f} steps={steps} done={done} "
                f"floor={obs.floor} rooms={obs.rooms} kills={obs.kills} time={obs.time:.2f}",
                flush=True,
            )

    fieldnames = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    scores = [float(row["score"]) for row in rows]
    rewards = [float(row["total_reward"]) for row in rows]
    print()
    print(f"output={args.output}")
    print(f"episodes={len(rows)} max_steps={args.max_steps} seed={args.seed}")
    print(
        "score "
        f"avg={statistics.mean(scores):.3f} "
        f"min={min(scores):.3f} "
        f"max={max(scores):.3f} "
        f"stdev={statistics.pstdev(scores):.3f}"
    )
    print(
        "reward "
        f"avg={statistics.mean(rewards):.3f} "
        f"min={min(rewards):.3f} "
        f"max={max(rewards):.3f}"
    )


if __name__ == "__main__":
    main()
