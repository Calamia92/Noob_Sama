from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dqn import DQNAgent
from src.eos_env import ACTIONS, EclipseEnv
from src.features import featurize
from src.policies import heuristic_action


DEFAULT_AGENT = ROOT / "models" / "dqn_agent.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload the saved DQN agent and watch it play in the browser."
    )
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--headless", action="store_true", help="Run without a visible window.")
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Synchronous fast-forward stepping (useful with --headless).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.02,
        help="Tiny exploration to break deterministic stuck loops (0 for pure greedy).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV file recording one row per episode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = DQNAgent.load(args.agent)
    print(
        f"agent={args.agent.name} features={agent.n_features} "
        f"actions={len(agent.actions)} episodes_trained={agent.episodes_trained}",
        flush=True,
    )
    scores: list[float] = []
    rows: list[dict[str, object]] = []

    with EclipseEnv(
        headless=args.headless,
        slow_mo=0,
        step_seconds=args.step_seconds,
        turbo=args.turbo,
    ) as env:
        for episode in range(1, args.episodes + 1):
            obs = env.reset()
            previous = None
            done = False
            steps = 0

            while not done and steps < args.max_steps:
                action = agent.act(featurize(obs, previous), epsilon=args.epsilon)
                if action == "heuristic":
                    resolved = heuristic_action(obs)
                    action = resolved if resolved in ACTIONS else "noop"
                previous = obs
                obs, _reward, done, _info = env.step(action)
                steps += 1

            score = env.get_score()
            scores.append(score)
            rows.append(
                {
                    "episode": episode,
                    "steps": steps,
                    "done": done,
                    "score": round(score, 6),
                    "floors": obs.floors,
                    "rooms": obs.rooms,
                    "kills": obs.kills,
                    "time": round(obs.time, 2),
                }
            )
            print(
                f"episode={episode} steps={steps} done={done} "
                f"score={score:.3f} floors={obs.floors} rooms={obs.rooms} "
                f"kills={obs.kills} time={obs.time:.2f}"
            )

    if args.output and rows:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    if scores:
        print(
            f"summary episodes={len(scores)} "
            f"avg={statistics.mean(scores):.3f} "
            f"min={min(scores):.3f} max={max(scores):.3f}"
        )


if __name__ == "__main__":
    main()
