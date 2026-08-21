from __future__ import annotations

import unittest
from dataclasses import replace

from scripts.train import train_reward
from src.agents import (
    CONCRETE_ACTIONS,
    SAFE_EXPLORATION_ACTIONS,
    TRAIN_ACTIONS,
    QLearningAgent,
    discretize,
    project_action,
)
from src.eos_env import Observation
from src.policies import heuristic_action, resolve_intent, should_delegate_action


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

        self.assertEqual(action, "right_shoot_left")

    def test_agent_action_migration_preserves_existing_values(self) -> None:
        agent = QLearningAgent(["noop", "up"])
        agent.q = {"s": [1.5, 2.5]}

        agent.set_actions(["up", "noop", "up_shoot_right"])

        self.assertEqual(agent.actions, ["up", "noop", "up_shoot_right"])
        self.assertEqual(agent.q["s"], [2.5, 1.5, 0.0])

    def test_safe_exploration_actions_are_train_actions(self) -> None:
        self.assertTrue(set(SAFE_EXPLORATION_ACTIONS).issubset(TRAIN_ACTIONS))
        self.assertIn("heuristic", SAFE_EXPLORATION_ACTIONS)
        self.assertNotIn("left_shoot_down_right", SAFE_EXPLORATION_ACTIONS)

    def test_heuristic_interacts_when_choice_is_in_range(self) -> None:
        action = heuristic_action(
            obs(near_choice={"x": 100, "y": 100, "distance": 20, "type": "item"})
        )

        self.assertEqual(action, "interact")

    def test_heuristic_avoids_risky_room_choice(self) -> None:
        action = heuristic_action(
            obs(
                room_type="altar",
                doors_open=True,
                near_choice={"x": 100, "y": 100, "distance": 20, "type": "item"},
                target_door={"x": 220, "y": 100, "distance": 120, "dir": "right"},
            )
        )

        self.assertEqual(action, "right")

    def test_heuristic_ignores_non_healing_pickup_during_combat(self) -> None:
        action = heuristic_action(
            obs(
                enemy_count=1,
                nearest_enemy={"x": 220, "y": 100, "distance": 120},
                pickups=[{"x": 110, "y": 100, "distance": 10, "type": "gold"}],
            )
        )

        self.assertNotEqual(action, "left")

    def test_heuristic_prioritizes_exit_over_pickup_after_combat(self) -> None:
        action = heuristic_action(
            obs(
                doors_open=True,
                target_door={"x": 220, "y": 100, "distance": 120, "dir": "right"},
                pickups=[{"x": 90, "y": 100, "distance": 10, "type": "gold"}],
            )
        )

        self.assertEqual(action, "right")

    def test_heuristic_strafes_while_shooting_in_crowded_combat(self) -> None:
        action = heuristic_action(
            obs(
                x=640,
                y=360,
                enemy_count=3,
                nearest_enemy={"x": 900, "y": 360, "distance": 260},
            )
        )

        self.assertIn(action, {"up_shoot_right", "down_shoot_right"})

    def test_heuristic_uses_diagonal_shot(self) -> None:
        action = heuristic_action(
            obs(
                x=640,
                y=360,
                enemy_count=3,
                nearest_enemy={"x": 900, "y": 150, "distance": 334},
            )
        )

        self.assertIn(action, {"up_shoot_up_right", "down_shoot_up_right"})

    def test_heuristic_uses_diagonal_dash_when_enemy_is_too_close(self) -> None:
        action = heuristic_action(
            obs(
                x=640,
                y=360,
                enemy_count=1,
                nearest_enemy={"x": 700, "y": 420, "distance": 85},
            )
        )

        self.assertEqual(action, "dash_up_left")

    def test_policy_delegates_when_room_is_clear(self) -> None:
        self.assertTrue(
            should_delegate_action(
                obs(
                    doors_open=True,
                    target_door={"x": 220, "y": 100, "distance": 120, "dir": "right"},
                ),
                "shoot_right",
            )
        )

    def test_policy_delegates_static_action_in_crowded_combat(self) -> None:
        self.assertTrue(
            should_delegate_action(
                obs(
                    enemy_count=3,
                    nearest_enemy={"x": 220, "y": 100, "distance": 120},
                ),
                "shoot_right",
            )
        )

    def test_exit_intent_targets_door(self) -> None:
        action = resolve_intent(
            obs(
                doors_open=True,
                target_door={"x": 220, "y": 100, "distance": 120, "dir": "right"},
            ),
            "exit",
        )

        self.assertEqual(action, "right")

    def test_kite_intent_moves_and_shoots(self) -> None:
        action = resolve_intent(
            obs(
                x=640,
                y=360,
                enemy_count=1,
                nearest_enemy={"x": 900, "y": 360, "distance": 260},
            ),
            "kite",
        )

        self.assertEqual(action, "left_shoot_right")

    def test_dash_away_intent_uses_directional_dash(self) -> None:
        action = resolve_intent(
            obs(
                x=640,
                y=360,
                enemy_count=1,
                nearest_enemy={"x": 700, "y": 420, "distance": 85},
            ),
            "dash_away",
        )

        self.assertEqual(action, "dash_up_left")

    def test_train_reward_rewards_progress_and_penalizes_damage(self) -> None:
        before = obs(hp=6, time=10.0)
        after = replace(before, hp=4, rooms=1, kills=2, time=12.0)

        self.assertAlmostEqual(train_reward(before, after, hp_penalty=2.0), 8.2)


if __name__ == "__main__":
    unittest.main()
