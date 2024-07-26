from time import perf_counter
from mushroom_rl.core.agent import Agent
import wandb
import os, sys
import numpy as np
import torch.random

from experiment_launcher import single_experiment, run_experiment
from mushroom_rl.core import Logger, Core, VectorCore
from mushroom_rl.utils.torch import TorchUtils
from mushroom_rl.utils.callbacks import CollectDataset

from spline_rl.utils.agent_builder import agent_builder
from spline_rl.utils.env_builder import env_builder

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

os.environ["WANDB_START_METHOD"] = "thread"


@single_experiment
def experiment(env: str = 'box_pushing',
               #group_name: str = "dummy",
               group_name: str = "base_run_bsmp_box_pushing",
               n_envs: int = 1,
               alg: str = "bsmp_eppo_box_pushing",
               n_epochs: int = 5000,
               n_episodes: int = 256,
               n_episodes_per_fit: int = 64,
               n_eval_episodes: int = 25,
               batch_size: int = 64,
               #n_episodes: int = 16,
               #n_episodes_per_fit: int = 4,
               #n_eval_episodes: int = 4,
               #batch_size: int = 4,
               use_cuda: bool = False,

               # agent params
               n_q_cps: int = 11,
               n_t_cps: int = 10,
               sigma_init_q: float = 1.0,
               sigma_init_t: float = 1.0,
               constraint_lr: float = 1e-2,
               mu_lr: float = 5e-5,
               value_lr: float = 5e-4,
               n_epochs_policy: int = 64,
               eps_ppo: float = 5e-2,
               #initial_entropy_lb: float = 118,
               #entropy_lb: float = -118,
               initial_entropy_lb: float = 52,
               entropy_lb: float = -52. / 2.,
               #initial_entropy_lb: float = 71,
               #entropy_lb: float = -71,
               entropy_lb_ep: int = 1000,
               t_scale: float = 1.0,
               q_scale: float = 1. / 50.,
               q_d_scale: float = 1. / 150.,
               q_dot_d_scale: float = 1. / 50.,
               q_ddot_d_scale: float = 1.0,

               # env params
               full_mass_matrix: bool = True,

               mode: str = "online",
               #mode: str = "disabled",
               seed: int = 444,
               quiet: bool = True,
               #render: bool = True,
               render: bool = False,
               results_dir: str = './logs',
               **kwargs):
    #if len(sys.argv) > 1:
    #    seed = int(sys.argv[1])
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_pts_fixed_begin = 1
    if "bsmp_eppo" in alg:
        n_pts_fixed_begin = 3
    n_pts_fixed_end = 0
    if "bsmp_eppo_box_pushing" == alg:
        n_pts_fixed_end = 2

    # TODO: add parameter regarding the constraint loss stuff
    agent_params = dict(
        alg=alg,
        seed=seed,
        n_dim=7,
        n_q_cps=n_q_cps,
        n_t_cps=n_t_cps,
        n_pts_fixed_begin=n_pts_fixed_begin,
        n_pts_fixed_end=n_pts_fixed_end,
        sigma_init_q=sigma_init_q,
        sigma_init_t=sigma_init_t,
        constraint_lr=constraint_lr,
        mu_lr=mu_lr,
        value_lr=value_lr,
        n_epochs_policy=n_epochs_policy,
        batch_size=batch_size,
        eps_ppo=eps_ppo,
        entropy_lb=entropy_lb,
        initial_entropy_lb=initial_entropy_lb,
        entropy_lb_ep=entropy_lb_ep,
        t_scale=t_scale,
        q_scale=q_scale,
        q_d_scale=q_d_scale,
        q_dot_d_scale=q_dot_d_scale,
        q_ddot_d_scale=q_ddot_d_scale,
    )

    name = (f"ePPO_{alg}_tdiv1qdiv50_150_10qdotscaled_50_"
            f"lr{agent_params['mu_lr']}_valuelr{agent_params['value_lr']}_bs{batch_size}_"
            f"constrlr{agent_params['constraint_lr']}_nep{n_episodes}_neppf{n_episodes_per_fit}_"
            f"neppol{agent_params['n_epochs_policy']}_epsppo{agent_params['eps_ppo']}_"
            f"sigmainit{agent_params['sigma_init_q']}q_{agent_params['sigma_init_t']}t_entlb{agent_params['entropy_lb']}_"
            f"entlbinit{agent_params['initial_entropy_lb']}_entlbep{agent_params['entropy_lb_ep']}_"
            f"nqcps{agent_params['n_q_cps']}_ntcps{agent_params['n_t_cps']}_{'fmm' if full_mass_matrix else 'dmm'}_seed{seed}")

    results_dir = os.path.join(results_dir, name)

    logger = Logger(log_name=env, results_dir=results_dir, seed=seed)

    if use_cuda:
        TorchUtils.set_default_device('cuda')

    run_params = dict(
        seed=seed,
        n_epochs=n_epochs,
        n_episodes=n_episodes,
        n_episodes_per_fit=n_episodes_per_fit,
        batch_size=batch_size,
    )

    env_params = dict(
        render_mode="human" if render else None,
        frame_skip = 1,
        full_mass_matrix = full_mass_matrix,
    )

    config = {**agent_params, **run_params, **env_params}

    wandb_run = wandb.init(project="corl24_box_pushing", config=config, dir=results_dir, name=name, entity="kicai",
              group=f'{group_name}', mode=mode)

    eval_params = dict(
        n_episodes=n_eval_episodes,
        quiet=quiet,
        render=render
    )

    env, env_info_ = env_builder(env, n_envs, env_params)
    env_info_['rl_info'].interpolation_order = 5

    agent = agent_builder(env_info_, agent_params)

    dataset_callback = CollectDataset()
    if n_envs > 1:
        core = VectorCore(agent, env, callbacks_fit=[dataset_callback])
    else:
        core = Core(agent, env, callbacks_fit=[dataset_callback])

    best_success = -np.inf
    best_J_det = -np.inf
    best_J_sto = -np.inf
    #if_learn = False
    if_learn = True
    for epoch in range(n_epochs):
        times = []
        times.append(perf_counter())
        print("Epoch: ", epoch)
        if if_learn:
            core.learn(n_episodes=n_episodes, n_episodes_per_fit=n_episodes_per_fit, quiet=quiet)
            print("Rs train: ", dataset_callback.get().undiscounted_return)
            print("Js train: ", dataset_callback.get().discounted_return)
            J_sto = np.mean(dataset_callback.get().discounted_return)
            init_states = dataset_callback.get().get_init_states()
            context = core.agent._context_builder(init_states)
            V_sto = np.mean(core.agent.value_function(context).detach().numpy())
            E = np.mean(core.agent.distribution.entropy(context).detach().numpy())
            VJ_bias = V_sto - J_sto
            constraints_violation_sto = core.agent.compute_constraint_losses(torch.stack(dataset_callback.get().theta_list, axis=0), context).detach().numpy()
            constraints_violation_sto_mean = np.mean(constraints_violation_sto, axis=0)
            constraints_violation_sto_max = np.max(constraints_violation_sto, axis=0)
            mu = core.agent.distribution.estimate_mu(context)
            constraints_violation_det = core.agent.compute_constraint_losses(mu, context).detach().numpy()
            constraints_violation_det_mean = np.mean(constraints_violation_det, axis=0)
            constraints_violation_det_max = np.max(constraints_violation_det, axis=0)
            q, q_dot, q_ddot, t, dt, duration = core.agent.policy.compute_trajectory_from_theta(mu, context)
            mean_duration = np.mean(duration.detach().numpy())
            dataset_callback.clean()
        else:
            J_sto = 0.
            V_sto = 0.
            E = 0.
            VJ_bias = 0.
            constraints_violation_sto_mean = np.zeros(18)
            constraints_violation_sto_max = np.zeros(18)
            constraints_violation_det_mean = np.zeros(18)
            constraints_violation_det_max = np.zeros(18)
            mean_duration = 0.

        times.append(perf_counter())
        # Evaluate
        J_det, R, success, states, actions, box_pos_dist, box_rot_dist, energy, episode_length, dataset_info = compute_metrics(core, eval_params)
        #assert False
        #wandb_plotting(core, states, actions, epoch)
        times.append(perf_counter())

        entropy_lb = np.maximum(agent_params["initial_entropy_lb"] +
            (agent_params["entropy_lb"] - agent_params["initial_entropy_lb"]) * epoch / agent_params["entropy_lb_ep"], agent_params["entropy_lb"])
        core.agent.distribution.set_e_lb(entropy_lb)

        if "logger_callback" in kwargs.keys():
            kwargs["logger_callback"](J_det, J_sto, V_sto, R, E, success)

        # Write logging
        logger.log_numpy(J_det=J_det, J_sto=J_sto, V_sto=V_sto, VJ_bias=VJ_bias, R=R, E=E,
                         success=success)
        logger.epoch_info(epoch, J_det=J_det, V_sto=V_sto, VJ_bias=VJ_bias, R=R, E=E,
                          success=success, box_pos_dist=box_pos_dist, box_rot_dist=box_rot_dist, energy=energy)
        wandb.log({
            "Reward/": {"J_det": J_det, "J_sto": J_sto, "V_sto": V_sto, "VJ_bias": VJ_bias, "R": R, "success": success},
            "Entropy/": {"E": E},
            "Constraints_sto/": {
                "avg/": {str(i): a for i, a in enumerate(constraints_violation_sto_mean)},
                "max/": {str(i): a for i, a in enumerate(constraints_violation_sto_max)}
            },
            "Constraints_det/": {
                "avg/": {str(i): a for i, a in enumerate(constraints_violation_det_mean)},
                "max/": {str(i): a for i, a in enumerate(constraints_violation_det_max)}
            },
            "Stats/": {
                "mean_duration": mean_duration,
                "box_pos_dist": box_pos_dist,
                "box_rot_dist": box_rot_dist,
                "energy": energy,
                "episode_length": episode_length,
            },
            "Constraints/": {
                "joint_pos": np.mean(dataset_info['joint_pos_constraint']),
                "joint_vel": np.mean(dataset_info['joint_vel_constraint']),
                "rod_tip_pos_constraint": np.mean(dataset_info['rod_tip_pos_constraint']),
                "qpos_constraint": np.mean(dataset_info['qpos_constraint']),
                "qvel_constraint": np.mean(dataset_info['qvel_constraint']),
            }                
        }, step=epoch)
        logger.info(f"BEST J_det: {best_J_det}")
        logger.info(f"BEST J_sto: {best_J_sto}")
        if hasattr(agent, "get_alphas"):
            wandb.log({
            "alphas/": {str(i): a for i, a in enumerate(agent.get_alphas())}
            }, step=epoch)

        if best_J_det <= J_det:
            best_J_det = J_det
            logger.log_agent(agent, epoch=epoch)
        
        if epoch % 100 == 0:
            logger.log_agent(agent, epoch=epoch)
        times.append(perf_counter())
        print("Epoch Times: ", times[1] - times[0], times[2] - times[1], times[3] - times[2])

    wandb_run.log_model(logger.path, name=f"{group_name}_{seed}")
    wandb_run.finish()



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
    current_idx = 0
    box_goal_pos_dist = []
    box_goal_rot_dist = []
    energy = []
    for episode_len in eps_length:
        success += dataset.info["success"][current_idx + episode_len - 1]
        box_goal_pos_dist.append(dataset.info["box_goal_pos_dist"][current_idx + episode_len - 1])
        box_goal_rot_dist.append(dataset.info["box_goal_rot_dist"][current_idx + episode_len - 1])
        energy.append(dataset.info["episode_energy"][current_idx + episode_len - 1])
        current_idx += episode_len
    success /= len(eps_length)

    state = dataset.state
    action = dataset.action

    return J, R, success, state, action, np.mean(box_goal_pos_dist), np.mean(box_goal_rot_dist), np.mean(energy), eps_length, dataset.info


if __name__ == "__main__":
    run_experiment(experiment)
