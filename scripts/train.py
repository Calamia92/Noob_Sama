from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import Error as PlaywrightError

from src.agents import TRAIN_ACTIONS, QLearningAgent, discretize
from src.eos_env import EclipseEnv, Observation


DEFAULT_OUTPUT = ROOT / "reports" / "training_scores.csv"
DEFAULT_BEST = ROOT / "models" / "best_agent.json"
DEFAULT_LATEST = ROOT / "checkpoints" / "latest.json"

FIELDNAMES = [
    "kind",
    "episode",
    "steps",
    "score",
    "total_reward",
    "epsilon",
    "q_states",
    "kills",
    "rooms",
    "floors",
    "done",
    "wall_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning agent.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=None,
        help="Per-episode decay. Default: reach epsilon-min at ~70%% of the run.",
    )
    parser.add_argument(
        "--hp-penalty",
        type=float,
        default=2.0,
        help="Training-only penalty per HP lost. The comparison score never changes.",
    )
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--best-path", type=Path, default=DEFAULT_BEST)
    parser.add_argument("--latest-path", type=Path, default=DEFAULT_LATEST)
    parser.add_argument("--resume", action="store_true", help="Resume from --latest-path.")
    return parser.parse_args()


def train_reward(before: Observation, after: Observation, hp_penalty: float) -> float:
    return (
        (after.floors - before.floors) * 100
        + (after.rooms - before.rooms) * 10
        + (after.kills - before.kills)
        + max(0.0, after.time - before.time) * 0.1
        - max(0.0, before.hp - after.hp) * hp_penalty
    )


def run_episode(
    env: EclipseEnv,
    agent: QLearningAgent,
    *,
    max_steps: int,
    greedy: bool,
    hp_penalty: float,
) -> dict[str, object]:
    started = time.monotonic()
    obs = env.reset()
    state = discretize(obs)
    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < max_steps:
        action = agent.act(state, greedy=greedy)
        previous = obs
        obs, _, done, _ = env.step(action)
        reward = train_reward(previous, obs, hp_penalty)
        next_state = discretize(obs)
        if not greedy:
            agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1

    return {
        "steps": steps,
        "score": round(EclipseEnv.compute_score(obs), 6),
        "total_reward": round(total_reward, 6),
        "kills": obs.kills,
        "rooms": obs.rooms,
        "floors": obs.floors,
        "done": done,
        "wall_seconds": round(time.monotonic() - started, 2),
    }


def run_episode_with_retries(
    env: EclipseEnv,
    agent: QLearningAgent,
    *,
    max_attempts: int = 3,
    **kwargs: object,
) -> dict[str, object]:
    for attempt in range(1, max_attempts + 1):
        try:
            return run_episode(env, agent, **kwargs)
        except (PlaywrightError, TimeoutError) as exc:
            print(
                f"warning: episode attempt {attempt} failed ({type(exc).__name__}), "
                "restarting browser",
                flush=True,
            )
            env.close()
            env.start()
    raise RuntimeError("Environment kept failing after repeated browser restarts.")


def main() -> None:
    args = parse_args()

    best_mean = float("-inf")
    start_episode = 1
    if args.resume and args.latest_path.exists():
        agent = QLearningAgent.load(args.latest_path, seed=args.seed)
        data = json.loads(args.latest_path.read_text(encoding="utf-8"))
        best_mean = data.get("best_mean", float("-inf"))
        start_episode = agent.episodes_trained + 1
        print(f"resuming at episode {start_episode} (epsilon={agent.epsilon:.3f})", flush=True)
    else:
        decay = args.epsilon_decay
        if decay is None:
            horizon = max(1.0, 0.7 * args.episodes)
            decay = (args.epsilon_min / args.epsilon_start) ** (1.0 / horizon)
        agent = QLearningAgent(
            TRAIN_ACTIONS,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=decay,
            seed=args.seed,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    append = args.resume and args.output.exists()
    csv_file = args.output.open("a" if append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if not append:
        writer.writeheader()

    def write_row(kind: str, episode: int, row: dict[str, object]) -> None:
        writer.writerow(
            {
                "kind": kind,
                "episode": episode,
                "epsilon": round(agent.epsilon, 4),
                "q_states": len(agent.q),
                **row,
            }
        )
        csv_file.flush()

    env = EclipseEnv(headless=True, step_seconds=args.step_seconds)
    env.start()
    try:
        for episode in range(start_episode, args.episodes + 1):
            row = run_episode_with_retries(
                env,
                agent,
                max_steps=args.max_steps,
                greedy=False,
                hp_penalty=args.hp_penalty,
            )
            write_row("train", episode, row)
            print(
                f"episode={episode:03d} score={row['score']:.3f} "
                f"reward={row['total_reward']:.3f} steps={row['steps']} "
                f"epsilon={agent.epsilon:.3f} states={len(agent.q)}",
                flush=True,
            )
            agent.decay_epsilon()
            agent.episodes_trained = episode
            agent.save(args.latest_path, extra={"best_mean": best_mean})

            if episode % args.eval_every == 0:
                scores: list[float] = []
                for _ in range(args.eval_episodes):
                    eval_row = run_episode_with_retries(
                        env,
                        agent,
                        max_steps=args.max_steps,
                        greedy=True,
                        hp_penalty=args.hp_penalty,
                    )
                    write_row("eval", episode, eval_row)
                    scores.append(float(eval_row["score"]))
                mean_score = statistics.mean(scores)
                print(f"eval@{episode}: mean={mean_score:.3f} best={best_mean:.3f}", flush=True)
                if mean_score > best_mean:
                    best_mean = mean_score
                    agent.save(args.best_path, extra={"best_eval_score": best_mean})
                    print(f"new best agent saved ({best_mean:.3f})", flush=True)
    finally:
        agent.save(args.latest_path, extra={"best_mean": best_mean})
        env.close()
        csv_file.close()

    print()
    print(f"output={args.output}")
    print(f"best={args.best_path} (mean eval score {best_mean:.3f})")


if __name__ == "__main__":
    main()
