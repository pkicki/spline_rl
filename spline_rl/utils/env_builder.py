from mushroom_rl.core import Logger, MultiprocessEnvironment

from envs.air_hockey_env import AirHockeyEnv


def env_builder(env_name, n_envs, env_params):
    env_class = None
    if env_name == "air_hockey":
        env_class = AirHockeyEnv
    else:
        raise ValueError("Unknown environment")
    
    if n_envs > 1:
        env = MultiprocessEnvironment(env_class, n_envs=n_envs, **env_params)
    elif n_envs == 1:
        env = env_class(**env_params)
    else:
        raise ValueError("n_envs must be greater than 0")
    return env, env.env_info
