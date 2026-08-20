from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eos_env import EclipseEnv
from src.policies import heuristic_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Eclipse Of Souls with a simple heuristic agent.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--watch", action="store_true", help="Open a visible browser window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores: list[float] = []

    with EclipseEnv(
        headless=not args.watch,
        slow_mo=50 if args.watch else 0,
        step_seconds=args.step_seconds,
    ) as env:
        for episode in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            steps = 0

            while not done and steps < args.max_steps:
                action = heuristic_action(obs)
                obs, reward, done, _info = env.step(action)
                steps += 1
                if args.watch:
                    print(
                        f"episode={episode} step={steps} action={action} "
                        f"reward={reward:.3f} score={env.get_score():.3f} "
                        f"room={obs.room_type} enemies={obs.enemy_count} "
                        f"pickups={obs.pickup_count} hp={obs.hp}",
                        flush=True,
                    )

            score = env.get_score()
            scores.append(score)
            print(
                f"episode={episode} steps={steps} done={done} "
                f"score={score:.3f} floor={obs.floor} rooms={obs.rooms} "
                f"kills={obs.kills} time={obs.time:.2f}"
            )

    if scores:
        print(
            f"summary episodes={len(scores)} "
            f"avg={sum(scores) / len(scores):.3f} "
            f"min={min(scores):.3f} max={max(scores):.3f}"
        )


if __name__ == "__main__":
    main()
