import argparse
import time
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eos_env import ACTIONS, EclipseEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for the game environment.")
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Drive the simulation synchronously and report the step rate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with EclipseEnv(headless=True, turbo=args.turbo) as env:
        obs = env.reset()
        print("initial:", obs)
        started = time.monotonic()
        steps = 0
        for action in ["up", "right", "shoot_right", "dash", "noop"] * (20 if args.turbo else 1):
            obs, reward, done, info = env.step(action)
            steps += 1
            if steps <= 5:
                print(action, "reward=", round(reward, 3), "done=", done, "info=", info)
            if done:
                break
        elapsed = time.monotonic() - started
        print(f"steps={steps} wall={elapsed:.2f}s rate={steps / elapsed:.1f} steps/s")
        print("score:", round(env.get_score(), 3))
        print("actions:", ", ".join(ACTIONS))


if __name__ == "__main__":
    main()
