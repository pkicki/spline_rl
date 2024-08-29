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
def experiment(env: str = 'bimanual',
               #group_name: str = "bsmp_bimanual_easy_pm10cm_newreward_betterinputscaling",
               group_name: str = "bsmp_bimanual_easypm10cm_newreward_betterinputscaling_onlyposabsorbing_drop02gripjoint",
               n_envs: int = 1,
               alg: str = "bsmp_eppo_bimanual",
               n_epochs: int = 5000,
               n_episodes: int = 256,
               n_episodes_per_fit: int = 64,
               n_eval_episodes: int = 10,
               batch_size: int = 64,
               #n_episodes: int = 64,
               #n_episodes_per_fit: int = 16,
               #n_eval_episodes: int = 2,
               #batch_size: int = 16,
               #n_episodes: int = 2,
               #n_episodes_per_fit: int = 2,
               #n_eval_episodes: int = 2,
               #batch_size: int = 2,
               use_cuda: bool = False,

               # agent params
               n_q_cps: int = 11,
               n_t_cps: int = 10,
               sigma_init_q: float = 1.0,
               sigma_init_t: float = 1.0,
               constraint_lr: float = 1e-2,
               mu_lr: float = 5e-5,
               value_lr: float = 5e-4,
               n_epochs_policy: int = 32,
               eps_ppo: float = 5e-2,
               #initial_entropy_lb: float = 118,
               #entropy_lb: float = -118,
               initial_entropy_lb: float = 88.,
               entropy_lb: float = -88. / 2.,
               #initial_entropy_lb: float = 71,
               #entropy_lb: float = -71,
               entropy_lb_ep: int = 500,
               t_scale: float = 1.0,
               q_scale: float = 1. / 1500.,
               q_d_scale: float = 1. / 500.,
               q_dot_d_scale: float = 1. / 50.,
               q_ddot_d_scale: float = 1.0,
               value_function_bias: float = 2.5,
               kl_threshold: float = 1e10,
               #cov: str = "diag",
               cov: str = "full",

               # env params
               gamma: float = 0.997,
               horizon: int = 400,
               full_mass_matrix: bool = True,
               interpolation_order: int = 5,

               debug: bool = False,
               #debug: bool = True,
               seed: int = 444,
               quiet: bool = True,
               render: bool = True,
               #render: bool = False,
               results_dir: str = './logs',
               **kwargs):
    #if len(sys.argv) > 1:
    #    seed = int(sys.argv[1])
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_pts_fixed_end = 0
    n_pts_fixed_begin = 1
    if "bsmp_eppo" in alg:
        n_pts_fixed_begin = 3
        n_pts_fixed_end = 2

    # TODO: add parameter regarding the constraint loss stuff
    agent_params = dict(
        env=env,
        alg=alg,
        seed=seed,
        n_dim=13,
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
        value_function_bias=value_function_bias,
        kl_threshold=kl_threshold,
        cov=cov,
    )

    name = (f"ePPO_bimanual_{alg}_tdiv1qdiv1500_500_easypm10cm_"
            f"lr{agent_params['mu_lr']}_valuelr{agent_params['value_lr']}_bs{batch_size}_"
            f"constrlr{agent_params['constraint_lr']}_nep{n_episodes}_neppf{n_episodes_per_fit}_"
            f"neppol{agent_params['n_epochs_policy']}_epsppo{agent_params['eps_ppo']}_"
            f"siginit{agent_params['sigma_init_q']}q_{agent_params['sigma_init_t']}t_entlb{agent_params['entropy_lb']}_"
            f"entlbinit{agent_params['initial_entropy_lb']}_entlbep{agent_params['entropy_lb_ep']}_klth{agent_params['kl_threshold']}_"
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
        gamma=gamma,
        horizon=horizon,
        render_mode="human" if render else None,
        interpolation_order=interpolation_order
    )

    config = {**agent_params, **run_params, **env_params}

    wandb_run = wandb.init(project="corl24_bimanual", config=config, dir=results_dir, name=name, entity="kicai",
              group=f'{group_name}', mode="online" if not debug else "disabled")

    eval_params = dict(
        n_episodes=n_eval_episodes,
        quiet=quiet,
        render=render
    )

    env, env_info_ = env_builder(env, n_envs, env_params)

    agent = agent_builder(env_info_, agent_params)

    ##agent_path = os.path.join(os.path.dirname(__file__), "logs/444/ePPO_bimanual_bsmp_eppo_bimanual_tdiv1qdiv1500_500_newreward_lr5e-05_valuelr0.0005_bs16_constrlr0.01_nep64_neppf16_neppol64_epsppo0.05_sigmainit1.0q_1.0t_entlb-44.0_entlbinit88.0_entlbep500_klth0.02_nqcps11_ntcps10_fmm_seed444/bimanual/agent-444-13.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "logs/444/ePPO_bimanual_bsmp_eppo_bimanual_tdiv1qdiv1500_500_newreward_onlyposabsorbing_lr5e-05_valuelr0.0005_bs16_constrlr0.01_nep64_neppf16_neppol64_epsppo0.05_sigmainit1.0q_1.0t_entlb-44.0_entlbinit88.0_entlbep500_klth0.02_nqcps11_ntcps10_fmm_seed444/bimanual/agent-444-15.msh")

    #print("Load agent from: ", agent_path)
    #agent = Agent.load(agent_path)
    #agent.load_constraints(env_info_['rl_info'])
    #agent._optimizer = torch.optim.Adam(agent.distribution.parameters(), lr=agent_params["mu_lr"])
    #agent.mdp_info = env_info_['rl_info']
    ##agent._epoch_no = 0
    #agent.task_losses = []
    #agent.scaled_constraint_losses = []
    #agent.task_losses = []
    #agent.last_kl_divergence = 0.
    #agent.kl_threshold = agent_params["kl_threshold"]
    ##agent.distribution._log_sigma_approximator.model.network._init_sigma *= 3.
    #agent.policy.t_scale = 1.
    #agent.policy.q_scale = 1. / 1500.
    #agent.policy.q_d_scale = 1. / 500.
    #agent.policy.q_dot_d_scale = 1. / 50.
    #agent.policy.q_ddot_d_scale = 1.
    #agent.policy._traj_no = 0

    dataset_callback = CollectDataset()
    if n_envs > 1:
        core = VectorCore(agent, env, callbacks_fit=[dataset_callback])
    else:
        core = Core(agent, env, callbacks_fit=[dataset_callback])

    best_success = -np.inf
    best_J_det = -np.inf
    best_J_sto = -np.inf
    #if_learn = False
    #if_learn = True
    for epoch in range(n_epochs):
        times = []
        times.append(perf_counter())
        print("Epoch: ", epoch)
        #if if_learn:
        #if True:
        if not debug:
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
        J_det, R, success, states, actions, goal_pos_dist, goal_rot_dist, left_handle_dist, right_handle_dist, \
                                                episode_length, dataset_info = compute_metrics(core, eval_params)
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
                          success=success, goal_pos_dist=goal_pos_dist, goal_rot_dist=goal_rot_dist)
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
                "goal_pos_dist": goal_pos_dist,
                "goal_rot_dist": goal_rot_dist,
                "left_handle_dist": left_handle_dist,
                "right_handle_dist": right_handle_dist,
                "episode_length": episode_length,
                "last_kl_divergence": core.agent.last_kl_divergence,
            },
            "Constraints/": {
                "joint_vel": np.mean(dataset_info['joint_vel_constraint']),
                "ee_dist_constraint": np.mean(dataset_info['ee_dist_constraint']),
                "left_distance_constraint": np.mean(dataset_info['left_distance_constraint']),
                "right_distance_constraint": np.mean(dataset_info['right_distance_constraint']),
                "orientation_constraint": np.mean(dataset_info['orientation_constraint']),
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
    goal_pos_dist = []
    goal_rot_dist = []
    left_handle_dist = []
    right_handle_dist = []
    print("EPOSIODES_LENGTHS:", eps_length)
    for episode_len in eps_length:
        success += dataset.info["success"][current_idx + episode_len - 1]
        goal_pos_dist.append(dataset.info["goal_pos_dist"][current_idx + episode_len - 1])
        goal_rot_dist.append(dataset.info["goal_rot_dist"][current_idx + episode_len - 1])
        left_handle_dist.append(np.mean(dataset.info["left_handle_dist"][current_idx:current_idx + episode_len - 1]))
        right_handle_dist.append(np.mean(dataset.info["right_handle_dist"][current_idx:current_idx + episode_len - 1]))
        #episode_left_ee_orientation = np.stack(dataset.info["left_ee_orientation"][current_idx:current_idx + episode_len - 1])
        #episode_right_ee_orientation = np.stack(dataset.info["right_ee_orientation"][current_idx:current_idx + episode_len - 1])
        current_idx += episode_len
    success /= len(eps_length)

    state = dataset.state
    action = dataset.action

    return J, R, success, state, action, np.mean(goal_pos_dist), np.mean(goal_rot_dist), \
           np.mean(left_handle_dist), np.mean(right_handle_dist), eps_length, dataset.info


if __name__ == "__main__":
    run_experiment(experiment)
