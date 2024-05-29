from experiments.air_hockey_deploy import load_env_agent
from experiments.air_hockey_episodic_exp import compute_metrics
import os
import numpy as np
import torch.random

from mushroom_rl.core import Core
from mushroom_rl.utils.callbacks import CollectDataset

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

os.environ["WANDB_START_METHOD"] = "thread"

torch.set_default_dtype(torch.float32)

def experiment(n_eval_episodes: int = 10,
               quiet: bool = True,
               render: bool = False,
               **kwargs):
    eval_params = dict(
        n_episodes=n_eval_episodes,
        quiet=quiet,
        render=render
    )

    agent_path = os.path.join(os.path.dirname(__file__), "trained_models/bsmp_eppo_stop_puzereward/agent-7-1382.msh")
    env, agent = load_env_agent(agent_path)


    dataset_callback = CollectDataset()
    core = Core(agent, env, callbacks_fit=[dataset_callback])

    J_det, R, success, states, actions, time_to_hit, max_puck_vel, episode_length, dataset_info = compute_metrics(core, eval_params)

    print("J_det:", J_det)
    print("R:", R)
    print("success:", success)
    print("time_to_hit:", time_to_hit)
    print("max_puck_vel:", max_puck_vel)
    print("episode_length:", episode_length)
    constraints = {
        "joint_pos": np.mean(dataset_info['joint_pos_constraint']),
        "joint_vel": np.mean(dataset_info['joint_vel_constraint']),
        "ee_xlb": np.mean(dataset_info['ee_xlb_constraint']),
        "ee_ylb": np.mean(dataset_info['ee_ylb_constraint']),
        "ee_yub": np.mean(dataset_info['ee_yub_constraint']),
        "ee_zlb": np.mean(dataset_info['ee_zlb_constraint']),
        "ee_zub": np.mean(dataset_info['ee_zub_constraint']),
        "ee_zeb": np.mean(dataset_info['ee_zeb_constraint']),
    }
    print("constraints:", constraints)


if __name__ == "__main__":
    experiment()
