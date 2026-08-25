from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.dqn import DQNAgent, ReplayBuffer
from src.eos_env import Observation
from src.features import N_FEATURES, featurize
from tests.test_core import obs


class FeatureTests(unittest.TestCase):
    def test_featurize_length_matches_constant(self) -> None:
        features = featurize(obs())
        self.assertEqual(len(features), N_FEATURES)

    def test_featurize_terminal_state_is_zero(self) -> None:
        features = featurize(obs(state="gameover"))
        self.assertEqual(features, [0.0] * N_FEATURES)

    def test_featurize_encodes_enemy_offset(self) -> None:
        rich = obs(
            x=640,
            y=360,
            enemy_count=1,
            nearest_enemy={"x": 900, "y": 360, "distance": 260},
            enemies=[
                {"x": 900, "y": 360, "distance": 260, "hp": 3, "max_hp": 4, "elite": False}
            ],
        )
        base = featurize(obs(x=640, y=360))
        features = featurize(rich)
        self.assertNotEqual(features, base)
        self.assertEqual(len(features), N_FEATURES)

    def test_featurize_flags_stuck_when_position_frozen(self) -> None:
        before = obs(x=200, y=200)
        features = featurize(obs(x=200, y=200), before)
        moved = featurize(obs(x=260, y=200), before)
        self.assertEqual(features[-1], 1.0)
        self.assertEqual(moved[-1], 0.0)


class DQNTests(unittest.TestCase):
    def test_learns_a_simple_target(self) -> None:
        agent = DQNAgent(4, ["a", "b"], hidden=(16, 16), gamma=0.0, lr=1e-2, seed=0)
        buffer = ReplayBuffer(64, 4)
        state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(64):
            buffer.add(state, 0, 1.0, state, True)
            buffer.add(state, 1, -1.0, state, True)
        for _ in range(300):
            agent.train_step(buffer.sample(32, agent.rng))
        q = agent.q_values(state)
        self.assertGreater(q[0], 0.8)
        self.assertLess(q[1], -0.8)

    def test_save_and_load_roundtrip(self) -> None:
        agent = DQNAgent(N_FEATURES, ["a", "b", "c"], seed=1)
        features = featurize(obs())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.json"
            agent.save(path)
            clone = DQNAgent.load(path)
        np.testing.assert_allclose(agent.q_values(features), clone.q_values(features), atol=1e-4)
        self.assertEqual(clone.actions, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
