from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dqn import DQNAgent
from src.eos_env import ACTIONS


DEFAULT_DEMOS = ROOT / "data" / "demos.npz"
DEFAULT_OUTPUT = ROOT / "models" / "bc_agent.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone the heuristic from recorded demonstrations."
    )
    parser.add_argument(
        "--demos",
        type=Path,
        nargs="+",
        default=[DEFAULT_DEMOS],
        help="One or more demo files (base demonstrations + DAgger rounds).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--init",
        type=Path,
        default=None,
        help="Warm-start from a saved agent instead of a fresh network.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_states = []
    all_actions = []
    for path in args.demos:
        data = np.load(path)
        all_states.append(data["states"])
        all_actions.append(data["actions"])
        print(f"loaded {len(data['actions'])} pairs from {path}")
    states = np.concatenate(all_states)
    actions = np.concatenate(all_actions)
    n_samples, n_features = states.shape
    print(f"demos: {n_samples} pairs, {n_features} features")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_samples)
    n_val = max(1, int(n_samples * args.val_split))
    val_idx, train_idx = order[:n_val], order[n_val:]

    if args.init:
        agent = DQNAgent.load(args.init, seed=args.seed)
        agent.lr = args.lr
        print(f"warm start from {args.init}")
    else:
        agent = DQNAgent(
            n_features,
            list(ACTIONS),
            hidden=tuple(args.hidden),
            lr=args.lr,
            seed=args.seed,
        )

    def accuracy(idx: np.ndarray) -> float:
        correct = 0
        for start in range(0, len(idx), 4096):
            batch = idx[start : start + 4096]
            logits = agent._forward(states[batch], agent.params)[-1]
            correct += int((logits.argmax(axis=1) == actions[batch]).sum())
        return correct / len(idx)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_idx)
        losses = []
        for start in range(0, len(train_idx), args.batch_size):
            batch = train_idx[start : start + args.batch_size]
            losses.append(agent.train_step_bc(states[batch], actions[batch]))
        val_acc = accuracy(val_idx)
        print(
            f"epoch={epoch:02d} loss={np.mean(losses):.4f} val_acc={val_acc:.3f}",
            flush=True,
        )
        if val_acc > best_val:
            best_val = val_acc
            agent.save(args.output, extra={"algo": "bc", "val_accuracy": best_val})

    print()
    print(f"best={args.output} (val accuracy {best_val:.3f})")


if __name__ == "__main__":
    main()
