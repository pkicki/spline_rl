from __future__ import annotations

import random
from time import perf_counter
from typing import Any, ClassVar

from spline_rl.envs.air_hockey_env import AirHockeyEnv
from spline_rl.envs.kinodynamic_cup_env import KinodynamicCupEnv
import torch
import numpy as np

from gymnasium import spaces

import omnisafe
from omnisafe.envs.core import CMDP, env_register
from omnisafe.evaluator import Evaluator

torch.set_default_dtype(torch.float32)

@env_register
class OmnisafeWrapper(CMDP):

    _support_envs: ClassVar[list[str]] = ['air_hockey', 'kinodynamic']
    # automatically reset when `terminated` or `truncated`
    need_auto_reset_wrapper = True
    # set `truncated=True` when the total steps exceed the time limit.
    need_time_limit_wrapper = False

    env_id_dict = {"air_hockey": AirHockeyEnv, "kinodynamic": KinodynamicCupEnv}

    def __init__(self, env_id: str, **kwargs: dict[str, Any]) -> None:
        self._count = 0
        self._num_envs = 1

        # self._device = kwargs['device']

        self._base_env = self.env_id_dict[env_id](return_cost=True)

        self._observation_space = spaces.Box(low=self._base_env.info.observation_space.low,
                                             high=self._base_env.info.observation_space.high)
        self.action_shape = self._base_env.info.action_space.shape
        self._action_space = spaces.Box(low=self._base_env.info.action_space.low.reshape(-1),
                                        high=self._base_env.info.action_space.high.reshape(-1))

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._count += 1

        action = action.reshape(self.action_shape)
        obs, reward, cost, terminated, info = self._base_env.step(action.cpu().numpy())

        obs = torch.as_tensor(obs).float()
        reward = torch.as_tensor(reward).float()
        cost = torch.as_tensor(np.maximum(cost, 0)).float()
        terminated = torch.as_tensor(terminated).float()
        truncated = torch.as_tensor(self._count > self.max_episode_steps)
        info["final_observation"] = obs

        return obs, reward, cost, terminated, truncated, info

    @property
    def max_episode_steps(self) -> int:
        """The max steps per episode."""
        return self._base_env.info.horizon

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        self.set_seed(seed)

        obs = self._base_env.reset()[0]
        obs = torch.as_tensor(obs).float()

        self._count = 0
        return obs, {}

    def set_seed(self, seed: int) -> None:
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def close(self) -> None:
        pass

    def render(self) -> Any:
        return self._base_env.render(record=True)


if __name__ == "__main__":
    custom_cfgs = {
        "algo_cfgs": {"steps_per_epoch": 256},
    }
    n_episodes = 256
    #n_episodes = 4
    avg_steps_per_episode = 65
    n_epochs = 5000
    custom_cfgs = {
        "algo_cfgs": {
            "steps_per_epoch": n_episodes * avg_steps_per_episode,
            'update_iters': 1,
        },
        "train_cfgs": {
            "device": "cpu",
            "total_steps": n_episodes * avg_steps_per_episode,
            "vector_env_nums": 1,
            "torch_threads": 1,
        },
    }
    agent = omnisafe.Agent(
        'PPOLag',
        'air_hockey',
        custom_cfgs=custom_cfgs,
    )
    t0 = perf_counter()
    agent.learn()
    t1 = perf_counter()
    print(t1 - t0)

