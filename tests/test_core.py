from __future__ import annotations

import unittest
from dataclasses import replace

from scripts.train import train_reward
from src.agents import CONCRETE_ACTIONS, discretize, project_action
from src.eos_env import Observation
from src.policies import heuristic_action


def obs(**overrides: object) -> Observation:
    base = Observation(
        state="play",
        hp=6,
        max_hp=6,
        x=100,
        y=100,
        floor=1,
        kills=0,
        rooms=0,
        floors=0,
        time=0.0,
        hp_missing=0,
        gold=0,
        item_count=0,
        room_type="start",
        room_cleared=False,
        abyss_gate=False,
        doors_open=False,
        portal_active=False,
        enemy_count=0,
        pickup_count=0,
        nearest_enemy=None,
        nearest_pickup=None,
        nearest_shop_item=None,
        nearest_door=None,
        target_door=None,
        near_choice=None,
        portal=None,
        available_doors=[],
        pickups=[],
    )
    return replace(base, **overrides)


class CoreBehaviorTests(unittest.TestCase):
    def test_discretize_encodes_enemy_direction_and_distance(self) -> None:
        state = discretize(
            obs(
                nearest_enemy={"x": 220, "y": 100, "distance": 120},
                enemy_count=1,
            )
        )

        self.assertEqual(state, "e:near:h0:d0:p0:i0")

    def test_discretize_encodes_clear_room_target_door(self) -> None:
        state = discretize(
            obs(
                doors_open=True,
                target_door={"x": 100, "y": 0, "distance": 100, "dir": "up"},
            )
        )

        self.assertEqual(state, "none:none:h0:dn:p0:i0")

    def test_project_action_maps_rich_combo_to_training_action(self) -> None:
        action = project_action("right_shoot_left", CONCRETE_ACTIONS, __import__("random").Random(1))

        self.assertIn(action, {"right", "shoot_left"})

    def test_heuristic_interacts_when_choice_is_in_range(self) -> None:
        action = heuristic_action(
            obs(near_choice={"x": 100, "y": 100, "distance": 20, "type": "item"})
        )

        self.assertEqual(action, "interact")

    def test_train_reward_rewards_progress_and_penalizes_damage(self) -> None:
        before = obs(hp=6, time=10.0)
        after = replace(before, hp=4, rooms=1, kills=2, time=12.0)

        self.assertAlmostEqual(train_reward(before, after, hp_penalty=2.0), 8.2)


if __name__ == "__main__":
    unittest.main()
