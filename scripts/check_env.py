from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eos_env import ACTIONS, EclipseEnv


def main() -> None:
    with EclipseEnv(headless=True) as env:
        obs = env.reset()
        print("initial:", obs)
        for action in ["up", "right", "shoot_right", "dash", "noop"]:
            obs, reward, done, info = env.step(action)
            print(action, "reward=", round(reward, 3), "done=", done, "obs=", obs, "info=", info)
        print("score:", round(env.get_score(), 3))
        print("actions:", ", ".join(ACTIONS))


if __name__ == "__main__":
    main()
