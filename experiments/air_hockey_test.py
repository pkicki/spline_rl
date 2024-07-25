from glob import glob
from time import perf_counter
from air_hockey_deploy import load_env_agent
import os
import sys
import numpy as np
import torch.random

from mushroom_rl.core import Core
from mushroom_rl.utils.callbacks import CollectDataset

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

os.environ["WANDB_START_METHOD"] = "thread"

#torch.set_default_dtype(torch.float32)

def compute_metrics(core, eval_params):
    with torch.no_grad():
        core.agent.set_deterministic(True)
        dataset = core.evaluate(**eval_params)
        core.agent.set_deterministic(False)

    J = dataset.discounted_return
    R = dataset.undiscounted_return

    eps_length = dataset.episodes_length
    current_idx = 0
    success = []
    time_to_hit = []
    max_puck_vel = []
    puck_poses = []
    scored = []
    joint_pos = []
    joint_vel = []
    ee_xlb = []
    ee_ylb = []
    ee_yub = []
    ee_zlb = []
    ee_zub = []
    ee_zeb = []
    for episode_len in eps_length:
        success.append(dataset.info["success"][current_idx + episode_len - 1])
        hit_time = dataset.info["hit_time"][current_idx + episode_len - 1]
        puck_poses.append(dataset.state[current_idx, :2])
        scored.append(dataset.info["success"][current_idx + episode_len - 1])

        if hit_time > 0:
            time_to_hit.append(hit_time)
        max_puck_vel.append(np.max(dataset.info["puck_velocity"][current_idx:current_idx + episode_len]))
        joint_pos.append(np.mean(dataset.info['joint_pos_constraint'][current_idx:current_idx+episode_len]))
        joint_vel.append(np.mean(dataset.info['joint_vel_constraint'][current_idx:current_idx+episode_len]))
        ee_xlb.append(np.mean(dataset.info['ee_xlb_constraint'][current_idx:current_idx+episode_len]))
        ee_ylb.append(np.mean(dataset.info['ee_ylb_constraint'][current_idx:current_idx+episode_len]))
        ee_yub.append(np.mean(dataset.info['ee_yub_constraint'][current_idx:current_idx+episode_len]))
        ee_zlb.append(np.mean(dataset.info['ee_zlb_constraint'][current_idx:current_idx+episode_len]))
        ee_zub.append(np.mean(dataset.info['ee_zub_constraint'][current_idx:current_idx+episode_len]))
        ee_zeb.append(np.mean(dataset.info['ee_zeb_constraint'][current_idx:current_idx+episode_len]))
        current_idx += episode_len

    return J, R, success, time_to_hit, max_puck_vel, eps_length, joint_pos, joint_vel, ee_xlb, ee_ylb, ee_yub, ee_zlb, ee_zub, ee_zeb


def experiment(n_eval_episodes: int = 100,#00,
               quiet: bool = True,
               render: bool = False,
               #render: bool = True,
               **kwargs):
    np.random.seed(444)
    eval_params = dict(
        n_episodes=n_eval_episodes,
        quiet=quiet,
        render=render
    )

    #model_type = "ours"
    #model_id = "7-1382"

    #model_type = "ours_unstructured"
    #model_id = "11-2140"

    model_type = "promp"
    #model_id = "0-1749"

    #model_type = "prodmp"
    #model_id = "11-2140"

    if len(sys.argv) > 1:
        model_type = sys.argv[1]

    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/bsmp_eppo_stop_puzereward/agent-7-1382.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/air_hockey/promp/agent-0-1749.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/air_hockey/ours_unstructured/agent-11-2140.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), f"trained_models/air_hockey/{model_type}/agent-{model_id}.msh")

    agent_paths = glob(os.path.join(os.path.dirname(__file__), f"trained_models/air_hockey_fixed/{model_type}/*.msh"))
    for agent_path in agent_paths:
        model_id = os.path.basename(agent_path).split(".")[0].replace("agent-", "")
        env, agent = load_env_agent(agent_path)

        if "promp" in agent_path:
            agent.policy.generate_basis()
        if "prodmp" in agent_path:
            agent.policy.generate_basis()
            agent.policy.N = agent.policy.N.astype(np.float64)
            agent.policy.dN = agent.policy.dN.astype(np.float64)

        dataset_callback = CollectDataset()
        core = Core(agent, env, callbacks_fit=[dataset_callback])

        t0 = perf_counter()
        (J_det, R, success, time_to_hit, max_puck_vel, episode_length, joint_pos, joint_vel, ee_xlb,
        ee_ylb, ee_yub, ee_zlb, ee_zub, ee_zeb) = compute_metrics(core, eval_params)
        t1 = perf_counter()
        print("elapsed time:", t1 - t0)

        results = dict(
            J_det=J_det,
            R=R,
            success=success,
            time_to_hit=time_to_hit,
            max_puck_vel=max_puck_vel,
            episode_length=episode_length,
            joint_pos_constraint=joint_pos,
            joint_vel_constraint=joint_vel,
            ee_xlb_constraint=ee_xlb,
            ee_ylb_constraint=ee_ylb,
            ee_yub_constraint=ee_yub,
            ee_zlb_constraint=ee_zlb,
            ee_zub_constraint=ee_zub,
            ee_zeb_constraint=ee_zeb
        )

        save_path = os.path.join(os.path.dirname(__file__), f"../paper/results/air_hockey_fixed/{model_type}")
        os.makedirs(save_path, exist_ok=True)
        np.savez(os.path.join(save_path, f"{model_id}.npz"), **results)


if __name__ == "__main__":
    experiment()
