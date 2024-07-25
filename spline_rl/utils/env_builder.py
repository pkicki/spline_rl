#from mushroom_rl.environments import Gymnasium
from mushroom_rl.environments.gymnasium_env import Gymnasium
from mushroom_rl.core import Logger, MultiprocessEnvironment
import fancy_gym

from spline_rl.envs.air_hockey_env import AirHockeyEnv
from spline_rl.envs.box_pushing_env import BoxPushingEnv
from spline_rl.envs.kinodynamic_cup_env import KinodynamicCupEnv


def env_builder(env_name, n_envs, env_params):
    env_class = None
    if env_name == "air_hockey":
        env_class = AirHockeyEnv
    elif env_name == "kinodynamic_cup":
        env_class = KinodynamicCupEnv
    elif env_name == "box_pushing":
        env_class = Gymnasium
        env_params["name"] = "fancy/BoxPushingConstrDensePDFF-v0"
        #env_class = BoxPushingEnv
    else:
        raise ValueError("Unknown environment")
    
    env = env_class(**env_params)
    env_info = env.env_info
    if n_envs > 1:
        env = MultiprocessEnvironment(env_class, n_envs=n_envs, **env_params)
    return env, env_info
