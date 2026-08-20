from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from src.eos_env import Observation, RANDOM_BASELINE_ACTIONS


CONCRETE_ACTIONS = [*RANDOM_BASELINE_ACTIONS, "interact"]

# The extra meta-action delegates one move to the rule-based policy; the
# Q-table then learns WHERE delegating beats its own moves.
TRAIN_ACTIONS = [*CONCRETE_ACTIONS, "heuristic"]

SECTORS = ["e", "ne", "n", "nw", "w", "sw", "s", "se"]

NEAR_ENEMY = 170.0
FAR_ENEMY = 430.0
INTERACT_RANGE = 58.0
PORTAL_RANGE = 65.0
PICKUP_RANGE = 400.0


def _sector8(dx: float, dy: float) -> str:
    angle = math.atan2(-dy, dx)
    index = int(round(angle / (math.pi / 4))) % 8
    return SECTORS[index]


def _useful_pickup_nearby(obs: Observation) -> bool:
    for pickup in obs.pickups:
        kind = pickup.get("type")
        if kind in {"heart", "heartHalf"} and obs.hp_missing <= 0:
            continue
        if kind not in {"heart", "heartHalf", "item", "gold"}:
            continue
        if pickup.get("distance", 9999.0) <= PICKUP_RANGE:
            return True
    return False


def _interactable_in_range(obs: Observation) -> bool:
    if obs.near_choice and obs.near_choice["distance"] <= INTERACT_RANGE:
        return True
    if obs.portal_active and obs.portal and obs.portal["distance"] <= PORTAL_RANGE:
        return True
    shop = obs.nearest_shop_item
    if shop and shop["distance"] <= INTERACT_RANGE and obs.gold >= shop.get("price", 0):
        return True
    return False


def discretize(obs: Observation) -> str:
    """Map a rich observation to a small state key for tabular learning."""
    if obs.state != "play":
        return "terminal"

    enemy = obs.nearest_enemy
    if enemy:
        direction = _sector8(enemy["x"] - obs.x, enemy["y"] - obs.y)
        if enemy["distance"] < NEAR_ENEMY:
            distance = "near"
        elif enemy["distance"] <= FAR_ENEMY:
            distance = "mid"
        else:
            distance = "far"
    else:
        direction = "none"
        distance = "none"

    hp_low = int(obs.hp * 2 <= obs.max_hp)
    # Once the room is clear, the agent needs to know WHERE the exit is,
    # otherwise every point of an empty room looks identical and it cannot
    # learn to leave (no rooms were ever cleared without this).
    # Relative 8-sector direction (not just the door's wall) so the agent can
    # learn to line up with the door frame before crossing it.
    exit_door = obs.target_door or obs.nearest_door
    if not obs.doors_open:
        doors = "d0"
    elif enemy is None and exit_door:
        doors = "d" + _sector8(exit_door["x"] - obs.x, exit_door["y"] - obs.y)
    else:
        doors = "d1"
    pickup = int(_useful_pickup_nearby(obs))
    interact = int(_interactable_in_range(obs))
    return f"{direction}:{distance}:h{hp_low}:{doors}:p{pickup}:i{interact}"


def project_action(action: str, actions: list[str], rng: random.Random) -> str:
    """Map a rich policy action (diagonals, free move+shoot combos) onto the
    training action set by picking one of its in-set components."""
    if action in actions:
        return action
    if "_shoot_" in action:
        move, _, shoot_dir = action.partition("_shoot_")
        candidates = [c for c in (move, "shoot_" + shoot_dir) if c in actions]
    else:
        candidates = [c for c in action.split("_") if c in actions]
    if candidates:
        return rng.choice(candidates)
    return "noop"


class QLearningAgent:
    def __init__(
        self,
        actions: list[str],
        *,
        alpha: float = 0.2,
        alpha_min: float = 0.05,
        alpha_decay: float = 1.0,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.99,
        seed: int | None = None,
    ) -> None:
        self.actions = list(actions)
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q: dict[str, list[float]] = {}
        self.episodes_trained = 0
        self.rng = random.Random(seed)

    def _q_values(self, state: str) -> list[float]:
        return self.q.setdefault(state, [0.0] * len(self.actions))

    def act(self, state: str, *, greedy: bool = False) -> str:
        qs = self._q_values(state)
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        best = max(qs)
        candidates = [i for i, value in enumerate(qs) if value == best]
        return self.actions[self.rng.choice(candidates)]

    def update(self, state: str, action: str, reward: float, next_state: str, done: bool) -> None:
        qs = self._q_values(state)
        index = self.actions.index(action)
        target = reward if done else reward + self.gamma * max(self._q_values(next_state))
        qs[index] += self.alpha * (target - qs[index])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # Alpha decays with it so late experience refines values instead of
        # overwriting them (the source of the post-peak oscillation).
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

    def save(self, path: Path | str, *, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "algo": "q_learning",
            "actions": self.actions,
            "alpha": self.alpha,
            "alpha_min": self.alpha_min,
            "alpha_decay": self.alpha_decay,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "episodes_trained": self.episodes_trained,
            "q": self.q,
        }
        if extra:
            payload.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str, *, seed: int | None = None) -> "QLearningAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            data["actions"],
            alpha=data["alpha"],
            alpha_min=data.get("alpha_min", 0.05),
            alpha_decay=data.get("alpha_decay", 1.0),
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
            seed=seed,
        )
        agent.q = {state: list(values) for state, values in data["q"].items()}
        agent.episodes_trained = data.get("episodes_trained", 0)
        return agent
