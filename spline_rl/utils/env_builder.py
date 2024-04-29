from mushroom_rl.core import Logger, MultiprocessEnvironment

from spline_rl.envs.air_hockey_env import AirHockeyEnv


def env_builder(env_name, n_envs, env_params):
    env_class = None
    if env_name == "air_hockey":
        env_class = AirHockeyEnv
    else:
        raise ValueError("Unknown environment")
    
    env = env_class(**env_params)
    env_info = env.env_info
    if n_envs > 1:
        env = MultiprocessEnvironment(env_class, n_envs=n_envs, **env_params)
    return env, env_info
