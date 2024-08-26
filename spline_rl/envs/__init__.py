from .air_hockey_env import AirHockeyEnv
from .kinodynamic_cup_env import KinodynamicCupEnv

from gymnasium.envs.registration import register as gymnasium_register
gymnasium_register(
    id="quad_insert_constrained",
    entry_point="spline_rl.envs.bimanual_env:BimanualEnv",
    max_episode_steps=1600,
)