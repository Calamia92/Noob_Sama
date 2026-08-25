from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import Error as PlaywrightError

from scripts.train import train_reward
from src.dqn import DQNAgent, ReplayBuffer
from src.eos_env import ACTIONS, EclipseEnv
from src.features import N_FEATURES, featurize
from src.policies import heuristic_action


DEFAULT_OUTPUT = ROOT / "reports" / "training_scores_dqn.csv"
DEFAULT_BEST = ROOT / "models" / "dqn_agent.json"
DEFAULT_LATEST = ROOT / "checkpoints" / "dqn_latest.json"

# Network-side reward scale only; the comparison score never changes.
REWARD_SCALE = 10.0

FIELDNAMES = [
    "kind",
    "episode",
    "steps",
    "score",
    "total_reward",
    "epsilon",
    "loss",
    "kills",
    "rooms",
    "floors",
    "done",
    "wall_seconds",
    "global_step",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN over the full action space.")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--step-seconds", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--target-sync", type=int, default=2_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=None,
        help="Linear decay horizon in env steps. Default: 60%% of the total budget.",
    )
    parser.add_argument(
        "--guide-ratio",
        type=float,
        default=0.25,
        help="Share of exploration steps played by the heuristic. Evaluation "
        "always uses the network alone.",
    )
    parser.add_argument("--hp-penalty", type=float, default=2.0)
    parser.add_argument("--door-shaping", type=float, default=0.02)
    parser.add_argument(
        "--time-reward",
        type=float,
        default=0.0,
        help="Weight of the survival-time term in the TRAINING reward. Zero "
        "by default: paying for idle time taught the network to camp.",
    )
    parser.add_argument(
        "--step-cost",
        type=float,
        default=0.02,
        help="Flat TRAINING penalty per step so standing still is never "
        "free (a frozen argmax loop otherwise costs nothing).",
    )
    parser.add_argument(
        "--eval-epsilon",
        type=float,
        default=0.02,
        help="Tiny exploration during greedy evals to break deterministic "
        "stuck loops, as in standard DQN evaluation protocols.",
    )
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--best-path", type=Path, default=DEFAULT_BEST)
    parser.add_argument("--latest-path", type=Path, default=DEFAULT_LATEST)
    parser.add_argument("--resume", action="store_true", help="Resume weights from --latest-path.")
    return parser.parse_args()


def run_episode(
    env: EclipseEnv,
    agent: DQNAgent,
    buffer: ReplayBuffer | None,
    rng: random.Random,
    args: argparse.Namespace,
    *,
    greedy: bool,
    global_step: int,
) -> tuple[dict[str, object], int]:
    started = time.monotonic()
    obs = env.reset()
    previous = None
    features = featurize(obs, previous)
    total_reward = 0.0
    losses: list[float] = []
    steps = 0
    done = False

    while not done and steps < args.max_steps:
        if greedy:
            action = agent.act(features, epsilon=args.eval_epsilon)
        elif rng.random() < epsilon_at(global_step, args):
            if rng.random() < args.guide_ratio:
                action = heuristic_action(obs)
                if action not in ACTIONS:
                    action = "noop"
            else:
                action = agent.actions[rng.randrange(len(agent.actions))]
        else:
            action = agent.act(features)

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
            if buffer.size >= args.warmup_steps and global_step % args.train_every == 0:
                losses.append(agent.train_step(buffer.sample(args.batch_size, agent.rng)))
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
        "loss": round(statistics.mean(losses), 6) if losses else "",
        "kills": obs.kills,
        "rooms": obs.rooms,
        "floors": obs.floors,
        "done": done,
        "wall_seconds": round(time.monotonic() - started, 2),
    }
    return row, global_step


def epsilon_at(global_step: int, args: argparse.Namespace) -> float:
    horizon = args.epsilon_decay_steps or max(1, int(0.6 * args.episodes * args.max_steps))
    progress = min(1.0, global_step / horizon)
    return args.epsilon_start + (args.epsilon_min - args.epsilon_start) * progress


def run_episode_with_retries(env: EclipseEnv, *pargs, max_attempts: int = 3, **kwargs):
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
    rng = random.Random(args.seed)
    actions = list(ACTIONS)

    best_mean = float("-inf")
    start_episode = 1
    if args.resume:
        agent = DQNAgent.load(args.latest_path, seed=args.seed)
        data = json.loads(args.latest_path.read_text(encoding="utf-8"))
        best_mean = data.get("best_mean", float("-inf"))
        start_episode = agent.episodes_trained + 1
        print(f"resuming at episode {start_episode}", flush=True)
    else:
        agent = DQNAgent(
            N_FEATURES,
            actions,
            hidden=tuple(args.hidden),
            gamma=args.gamma,
            lr=args.lr,
            seed=args.seed,
        )
    buffer = ReplayBuffer(args.buffer_size, N_FEATURES)
    global_step = agent.steps_trained

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
                "epsilon": round(epsilon_at(global_step, args), 4),
                "global_step": global_step,
                **row,
            }
        )
        csv_file.flush()

    env = EclipseEnv(headless=True, turbo=True, step_seconds=args.step_seconds)
    env.start()
    try:
        for episode in range(start_episode, args.episodes + 1):
            row, global_step = run_episode_with_retries(
                env, agent, buffer, rng, args, greedy=False, global_step=global_step
            )
            write_row("train", episode, row)
            print(
                f"episode={episode:03d} score={row['score']:.3f} "
                f"reward={row['total_reward']:.3f} steps={row['steps']} "
                f"floors={row['floors']} rooms={row['rooms']} kills={row['kills']} "
                f"eps={epsilon_at(global_step, args):.3f} loss={row['loss']} "
                f"wall={row['wall_seconds']}s",
                flush=True,
            )
            agent.episodes_trained = episode
            agent.steps_trained = global_step
            agent.save(args.latest_path, extra={"best_mean": best_mean})

            if episode % args.eval_every == 0:
                scores: list[float] = []
                floors: list[int] = []
                for _ in range(args.eval_episodes):
                    eval_row, global_step = run_episode_with_retries(
                        env, agent, None, rng, args, greedy=True, global_step=global_step
                    )
                    write_row("eval", episode, eval_row)
                    scores.append(float(eval_row["score"]))
                    floors.append(int(eval_row["floors"]))
                mean_score = statistics.mean(scores)
                print(
                    f"eval@{episode}: mean={mean_score:.3f} floors={sum(floors)} "
                    f"best={best_mean:.3f}",
                    flush=True,
                )
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
