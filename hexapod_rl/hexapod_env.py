import gymnasium as gym
import numpy as np

class HexapodEnv(gym.Env):
    def __init__(self, size: int = 5):
        self.size = size
        self._agent_location = np.array([-1,-1], dtype=np.int32)
        self._taget_location = np.array([-1,-1], dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space = gym.spaces.Discrete(4)
        self_action_to_direction = {
            0: np.array([0,1]),
            1: np.array([-1,0]),
            2: np.array([0,-1]),
            3: np.array([1,0]),
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    