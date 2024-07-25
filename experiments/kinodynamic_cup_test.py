from glob import glob
import sys
import os
import numpy as np
import torch.random

#from experiments.kinodynamic_deploy import load_env_agent
from kinodynamic_deploy import load_env_agent
from mushroom_rl.core import Core
from mushroom_rl.utils.callbacks import CollectDataset

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

os.environ["WANDB_START_METHOD"] = "thread"

#torch.autograd.set_detect_anomaly(True)
#torch.set_default_dtype(torch.float32)

def compute_metrics(core, eval_params):
    with torch.no_grad():
        core.agent.set_deterministic(True)
        dataset = core.evaluate(**eval_params)
        core.agent.set_deterministic(False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    print("Rs val:", dataset.undiscounted_return)
    print("Js val:", dataset.discounted_return)

    eps_length = dataset.episodes_length
    success = 0
    success_position = 0
    success_orientation = 0
    success_velocity = 0
    current_idx = 0
    joint_pos = []
    joint_vel = []
    orientation = []
    collision = []
    for episode_len in eps_length:
        #success += dataset.info["success"][current_idx + episode_len - 1]
        #success_position += dataset.info["success_position"][current_idx + episode_len - 1]
        #success_orientation += dataset.info["success_orientation"][current_idx + episode_len - 1]
        #success_velocity += dataset.info["success_velocity"][current_idx + episode_len - 1]
        success += np.mean(dataset.info["success"][current_idx:current_idx + episode_len])
        success_position += np.mean(dataset.info["success_position"][current_idx:current_idx + episode_len - 1])
        success_orientation += np.mean(dataset.info["success_orientation"][current_idx:current_idx + episode_len - 1])
        success_velocity += np.mean(dataset.info["success_velocity"][current_idx:current_idx + episode_len - 1])
        joint_pos.append(np.mean(dataset.info['joint_pos_constraint'][current_idx:current_idx+episode_len]))
        joint_vel.append(np.mean(dataset.info['joint_vel_constraint'][current_idx:current_idx+episode_len]))
        orientation.append(np.mean(dataset.info['orientation_constraint'][current_idx:current_idx+episode_len]))
        collision.append(np.mean(dataset.info['collision_constraint'][current_idx:current_idx+episode_len]))
        current_idx += episode_len
    success /= len(eps_length)
    success_position /= len(eps_length)
    success_orientation /= len(eps_length)
    success_velocity /= len(eps_length)

    return J, R, success, success_position, success_orientation, success_velocity, eps_length, joint_pos, joint_vel, orientation, collision

def experiment(n_eval_episodes: int = 100,
               seed: int = 444,
               quiet: bool = True,
               render: bool = False,
               #render: bool = True,
               **kwargs):
    np.random.seed(seed)
    torch.manual_seed(seed)

    #model_type = "promp_unstructured"
    #model_type = "prodmp_structured"
    model_type = "ours_unstructured"
    model_type = sys.argv[1] if len(sys.argv) > 1 else model_type

    eval_params = dict(
        n_episodes=n_eval_episodes,
        quiet=quiet,
        render=render
    )

    n_seeds = 1
    for i in range(n_seeds):
        agent_paths = glob(os.path.join(os.path.dirname(__file__), f"trained_models/kinodynamic/{model_type}/agent-{i}-2*.msh"))
        if not agent_paths:
            continue
        agent_path = sorted(agent_paths)[-1]
        model_id = os.path.basename(agent_path).split(".")[0].replace("agent-", "")
        if "unstructured" in model_type:
            q_d_scale = 1. / 50.
        else:
            q_d_scale = 1. / 150.
        env, agent = load_env_agent(agent_path, q_d_scale=q_d_scale)

        dataset_callback = CollectDataset()
        core = Core(agent, env, callbacks_fit=[dataset_callback])

        J_det, R, success, success_position, success_orientation, success_velocity, \
        eps_length, joint_pos, joint_vel, orientation, collision = compute_metrics(core, eval_params)

        results = dict(
                J_det=J_det,
                R=R,
                success=success,
                joint_pos_constraint=joint_pos,
                joint_vel_constraint=joint_vel,
                orientation_constraint=orientation,
                collision_constraint=collision,
            )
        print(results)

        assert False
        save_path = os.path.join(os.path.dirname(__file__), f"../paper/results/kino/{model_type}")
        os.makedirs(save_path, exist_ok=True)
        np.savez(os.path.join(save_path, f"{model_id}.npz"), **results)


if __name__ == "__main__":
    experiment()
