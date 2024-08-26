#from mushroom_rl.environments import Gymnasium
from mushroom_rl.environments.gymnasium_env import Gymnasium
from mushroom_rl.core import Logger, MultiprocessEnvironment
import fancy_gym

from spline_rl.envs import AirHockeyEnv, KinodynamicCupEnv
from spline_rl.envs.bimanual_env import BimanualEnv


def env_builder(env_name, n_envs, env_params):
    env_class = None
    if env_name == "air_hockey":
        env_class = AirHockeyEnv
    elif env_name == "kinodynamic_cup":
        env_class = KinodynamicCupEnv
    elif env_name == "box_pushing":
        env_class = Gymnasium
        env_params["name"] = "fancy/BoxPushingConstrDensePDFF-v0"
    elif env_name == "bimanual":
        env_class = BimanualEnv
    else:
        raise ValueError("Unknown environment")
    
    env = env_class(**env_params)
    env_info = env.env_info
    if n_envs > 1:
        env = MultiprocessEnvironment(env_class, n_envs=n_envs, **env_params)
    return env, env_info
