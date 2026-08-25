from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents import CONCRETE_ACTIONS, project_action
from src.dqn import DQNAgent
from src.eos_env import ACTIONS, EclipseEnv
from src.features import featurize
from src.policies import heuristic_action


DEFAULT_OUTPUT = ROOT / "data" / "demos.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record heuristic demonstrations as (features, action) pairs."
    )
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="DAgger mode: this saved agent PLAYS the episodes while the "
        "heuristic only provides the recorded labels, so the dataset "
        "covers the states the clone actually reaches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    action_index = {name: i for i, name in enumerate(ACTIONS)}
    rng = random.Random(args.seed)
    policy = DQNAgent.load(args.policy) if args.policy else None
    states: list[list[float]] = []
    actions: list[int] = []
    scores: list[float] = []
    floors_total = 0

    with EclipseEnv(headless=True, turbo=True, step_seconds=args.step_seconds) as env:
        for episode in range(1, args.episodes + 1):
            obs = env.reset()
            previous = None
            done = False
            steps = 0
            while not done and steps < args.max_steps:
                features = featurize(obs, previous)
                label = project_action(heuristic_action(obs), CONCRETE_ACTIONS, rng)
                states.append(features)
                actions.append(action_index[label])
                if policy is not None:
                    action = policy.act(features, epsilon=0.02)
                    if action not in ACTIONS:
                        action = label
                else:
                    action = label
                previous = obs
                obs, _, done, _ = env.step(action)
                steps += 1
            scores.append(env.get_score())
            floors_total += obs.floors
            if episode % 10 == 0:
                print(
                    f"episode={episode} pairs={len(states)} floors_total={floors_total}",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int32),
    )
    print(f"saved {len(states)} pairs to {args.output}")
    print(
        f"episodes={args.episodes} avg_score={statistics.mean(scores):.2f} "
        f"floors={floors_total}"
    )


if __name__ == "__main__":
    main()
