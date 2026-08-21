from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents import CONCRETE_ACTIONS, QLearningAgent, discretize, project_action
from src.eos_env import EclipseEnv
from src.policies import heuristic_action, resolve_intent, should_delegate_action


DEFAULT_AGENT = ROOT / "models" / "best_agent.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload the saved agent and watch it play in the browser."
    )
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--headless", action="store_true", help="Run without a visible window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = QLearningAgent.load(args.agent)
    print(
        f"agent={args.agent.name} states={len(agent.q)} "
        f"episodes_trained={agent.episodes_trained}",
        flush=True,
    )
    scores: list[float] = []

    with EclipseEnv(
        headless=args.headless,
        slow_mo=0 if args.headless else 50,
        step_seconds=args.step_seconds,
    ) as env:
        rng = random.Random()
        for episode in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            steps = 0
            trail: list[tuple[float, float]] = []

            while not done and steps < args.max_steps:
                action = agent.act(discretize(obs), greedy=True)
                if should_delegate_action(obs, action):
                    action = "heuristic"
                # A deterministic greedy policy can pin itself against an
                # obstacle the state does not encode; delegate that recovery
                # to the heuristic so it still aims for useful exits/pickups.
                trail.append((obs.x, obs.y))
                if len(trail) > 8:
                    trail.pop(0)
                    spread_x = max(p[0] for p in trail) - min(p[0] for p in trail)
                    spread_y = max(p[1] for p in trail) - min(p[1] for p in trail)
                    if spread_x < 12 and spread_y < 12 and obs.enemy_count == 0:
                        action = project_action(heuristic_action(obs), CONCRETE_ACTIONS, rng)
                        trail.clear()
                if action == "heuristic":
                    action = resolve_intent(obs, action)
                else:
                    action = resolve_intent(obs, action)
                action = project_action(action, CONCRETE_ACTIONS, rng)
                obs, _reward, done, _info = env.step(action)
                steps += 1
                if not args.headless:
                    print(
                        f"episode={episode} step={steps} action={action} "
                        f"score={env.get_score():.3f} hp={obs.hp} "
                        f"room={obs.room_type} enemies={obs.enemy_count}",
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
            f"avg={statistics.mean(scores):.3f} "
            f"min={min(scores):.3f} max={max(scores):.3f}"
        )


if __name__ == "__main__":
    main()
