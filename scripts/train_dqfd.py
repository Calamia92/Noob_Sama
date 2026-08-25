from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import Error as PlaywrightError

from scripts.train import train_reward
from src.dqn import DQNAgent, ReplayBuffer
from src.eos_env import EclipseEnv
from src.features import N_FEATURES, featurize


DEFAULT_DEMOS = [
    ROOT / "data" / "demos.npz",
    ROOT / "data" / "dagger_a.npz",
    ROOT / "data" / "dagger_b.npz",
]
DEFAULT_OUTPUT = ROOT / "reports" / "training_dqfd.csv"
DEFAULT_BEST = ROOT / "checkpoints" / "dqfd_best.json"

REWARD_SCALE = 10.0

FIELDNAMES = [
    "kind",
    "episode",
    "steps",
    "score",
    "total_reward",
    "td_loss",
    "bc_loss",
    "kills",
    "rooms",
    "floors",
    "done",
    "wall_seconds",
    "global_step",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL fine-tuning from demonstrations: TD loss on fresh "
        "experience plus a supervised anchor on the demo dataset, starting "
        "from a cloned agent."
    )
    parser.add_argument("--agent", type=Path, default=ROOT / "models" / "bc_agent.json")
    parser.add_argument("--demos", type=Path, nargs="+", default=DEFAULT_DEMOS)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--eval-epsilon", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bc-batch", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--warmup-steps", type=int, default=3_000)
    parser.add_argument("--target-sync", type=int, default=2_000)
    parser.add_argument("--hp-penalty", type=float, default=1.0)
    parser.add_argument("--door-shaping", type=float, default=0.02)
    parser.add_argument("--step-cost", type=float, default=0.02)
    parser.add_argument("--time-reward", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=15)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--best-path", type=Path, default=DEFAULT_BEST)
    return parser.parse_args()


def run_episode(env, agent, buffer, demos, args, *, greedy: bool, global_step: int):
    started = time.monotonic()
    obs = env.reset()
    previous = None
    features = featurize(obs, previous)
    total_reward = 0.0
    td_losses: list[float] = []
    bc_losses: list[float] = []
    steps = 0
    done = False
    demo_states, demo_actions = demos

    while not done and steps < args.max_steps:
        epsilon = args.eval_epsilon if greedy else args.epsilon
        action = agent.act(features, epsilon=epsilon)
        next_obs, _, done, _ = env.step(action)
        reward = (
            train_reward(
                obs,
                next_obs,
                args.hp_penalty,
                args.door_shaping,
                time_weight=args.time_reward,
            )
            - args.step_cost
        ) / REWARD_SCALE
        next_features = featurize(next_obs, obs)

        if buffer is not None:
            buffer.add(features, agent.actions.index(action), reward, next_features, done)
            global_step += 1
            if buffer.size >= args.warmup_steps:
                td_losses.append(agent.train_step(buffer.sample(args.batch_size, agent.rng)))
                idx = agent.rng.integers(0, len(demo_actions), size=args.bc_batch)
                bc_losses.append(agent.train_step_bc(demo_states[idx], demo_actions[idx]))
            if global_step % args.target_sync == 0:
                agent.sync_target()

        previous = obs
        obs = next_obs
        features = next_features
        total_reward += reward
        steps += 1

    row = {
        "steps": steps,
        "score": round(EclipseEnv.compute_score(obs), 6),
        "total_reward": round(total_reward, 6),
        "td_loss": round(statistics.mean(td_losses), 6) if td_losses else "",
        "bc_loss": round(statistics.mean(bc_losses), 6) if bc_losses else "",
        "kills": obs.kills,
        "rooms": obs.rooms,
        "floors": obs.floors,
        "done": done,
        "wall_seconds": round(time.monotonic() - started, 2),
    }
    return row, global_step


def run_episode_with_retries(env, *pargs, max_attempts: int = 3, **kwargs):
    for attempt in range(1, max_attempts + 1):
        try:
            return run_episode(env, *pargs, **kwargs)
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
    agent = DQNAgent.load(args.agent, seed=args.seed)
    agent.lr = args.lr
    agent.gamma = args.gamma
    agent.sync_target()
    print(f"fine-tuning from {args.agent} ({agent.n_features} features)")

    all_states = []
    all_actions = []
    for path in args.demos:
        data = np.load(path)
        all_states.append(data["states"])
        all_actions.append(data["actions"])
    demo_states = np.concatenate(all_states)
    demo_actions = np.concatenate(all_actions)
    print(f"demo anchor: {len(demo_actions)} pairs")

    buffer = ReplayBuffer(args.buffer_size, N_FEATURES)
    global_step = 0
    best_mean = float("-inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_file = args.output.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    writer.writeheader()

    def write_row(kind: str, episode: int, row: dict[str, object]) -> None:
        writer.writerow({"kind": kind, "episode": episode, "global_step": global_step, **row})
        csv_file.flush()

    env = EclipseEnv(headless=True, turbo=True, step_seconds=args.step_seconds)
    env.start()
    try:
        for episode in range(1, args.episodes + 1):
            row, global_step = run_episode_with_retries(
                env,
                agent,
                buffer,
                (demo_states, demo_actions),
                args,
                greedy=False,
                global_step=global_step,
            )
            write_row("train", episode, row)
            print(
                f"episode={episode:03d} score={row['score']:.3f} steps={row['steps']} "
                f"floors={row['floors']} rooms={row['rooms']} kills={row['kills']} "
                f"td={row['td_loss']} bc={row['bc_loss']}",
                flush=True,
            )

            if episode % args.eval_every == 0:
                scores = []
                for _ in range(args.eval_episodes):
                    eval_row, global_step = run_episode_with_retries(
                        env,
                        agent,
                        None,
                        (demo_states, demo_actions),
                        args,
                        greedy=True,
                        global_step=global_step,
                    )
                    write_row("eval", episode, eval_row)
                    scores.append(float(eval_row["score"]))
                mean_score = statistics.mean(scores)
                print(f"eval@{episode}: mean={mean_score:.3f} best={best_mean:.3f}", flush=True)
                if mean_score > best_mean:
                    best_mean = mean_score
                    agent.save(args.best_path, extra={"best_eval_score": best_mean})
                    print(f"new best saved ({best_mean:.3f})", flush=True)
    finally:
        env.close()
        csv_file.close()

    print()
    print(f"best={args.best_path} (mean eval score {best_mean:.3f})")


if __name__ == "__main__":
    main()
