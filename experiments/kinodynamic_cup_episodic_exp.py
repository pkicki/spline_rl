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

#torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

@single_experiment
def experiment(env: str = 'kinodynamic_cup',
               group_name: str = "test_single",
               n_envs: int = 1,
               alg: str = "bsmp_eppo_kinodynamic",
               #alg: str = "prodmp_eppo_unstructured",
               n_epochs: int = 2000,
               #n_episodes: int = 256,
               #n_episodes_per_fit: int = 64,
               #n_eval_episodes: int = 25,
               #batch_size: int = 64,
               n_episodes: int = 16,
               n_episodes_per_fit: int = 16,
               n_eval_episodes: int = 2,
               batch_size: int = 16,
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
               initial_entropy_lb: float = 45,
               entropy_lb: float = -45,
               entropy_lb_ep: int = 500,
               t_scale: float = 1.0,
               q_scale: float = 1. / 50.,
               q_d_scale: float = 1. / 150.,
               q_dot_d_scale: float = 1. / 50.,
               q_ddot_d_scale: float = 1.0,

               # env params
               horizon: int = 100,
               gamma: float = 0.99,

               mode: str = "online",
               #mode: str = "disabled",
               seed: int = 444,
               quiet: bool = True,
               #render: bool = False,
               render: bool = True,
               results_dir: str = './logs',
               **kwargs):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # TODO: add parameter regarding the constraint loss stuff
    agent_params = dict(
        alg=alg,
        seed=seed,
        n_dim=7,
        n_q_cps=n_q_cps,
        n_t_cps=n_t_cps,
        n_pts_fixed_begin=3 if "bsmp_eppo" in alg else 1,
        n_pts_fixed_end=3 if "bsmp_eppo" in alg else 0,
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

    name = (f"ePPO_kino_tdiv1qdiv50_150_"
            f"gamma099_hor150_"
            f"lr{agent_params['mu_lr']}_valuelr{agent_params['value_lr']}_bs{batch_size}_"
            f"constrlr{agent_params['constraint_lr']}_nep{n_episodes}_neppf{n_episodes_per_fit}_"
            f"neppol{agent_params['n_epochs_policy']}_epsppo{agent_params['eps_ppo']}_"
            f"sigmainit{agent_params['sigma_init_q']}q_{agent_params['sigma_init_t']}t_entlb{agent_params['entropy_lb']}_"
            f"entlbinit{agent_params['initial_entropy_lb']}_entlbep{agent_params['entropy_lb_ep']}_"
            f"nqcps{agent_params['n_q_cps']}_ntcps{agent_params['n_t_cps']}_seed{seed}")

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
        horizon=horizon,
        gamma=gamma,
        interpolation_order=3 if "prodmp" in alg else 5,
    )

    config = {**agent_params, **run_params, **env_params}

    wandb_run = wandb.init(project="corl24_experiments_kinodynamic", config=config, dir=results_dir, name=name, entity="kicai",
              group=f'{group_name}', mode=mode)

    eval_params = dict(
        n_episodes=n_eval_episodes,
        quiet=quiet,
        render=render
    )

    env, env_info_ = env_builder(env, n_envs, env_params)

    agent = agent_builder(env_info_, agent_params)

    # TRO hit moving with stop
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_stop_verynewreward_gamma2_interpm1_tdiv1qdiv50_150_10qdotscaled_50_gamma099_hor150_lr5e-05_valuelr0.0005_bs64_constrlr0.01_nep256_neppf64_neppol32_epsppo0.05_sigmainit1.0q_1.0t_entlb-118_entlbinit118_entlbep500_nqcps11_ntcps10_seed1/agent-1-571.msh")

    # unstructured moving
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_unstructured_gauss2_end131_yrangem035moving_qdiv50tdiv5_gamma099_hor150_lr5e-05_valuelr0.0005_bs64_constrlr0.01_nep64_neppf64_neppol32_epsppo0.05_sigmainit1.0q_1.0t_entlb-52_entlbinit52_entlbep6000_nqcps11_ntcps10_seed444/agent-444-3856.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_unstructured_gauss1_yrangem035moving_end131_qdiv50t5_gamma099_hor150_lr5e-05_valuelr0.0005_bs64_constrlr0.01_nep64_neppf64_neppol32_epsppo0.02_entlb-52_entlbinit52_entlbep6000_nqcps11_ntcps10_seed444/agent-444-3156.msh")
    
    # TRO hit moving
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_gamma2_interpm1_tdiv1qdiv50_150_10qdotscaled_50_gamma099_hor150_lr5e-05_valuelr0.0005_bs64_constrlr0.01_nep64_neppf64_neppol32_epsppo0.05_sigmainit1.0q_1.0t_entlb-52_entlbinit52_entlbep2000_nqcps11_ntcps10_seed444/agent-444-1207.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_TROhitmoving_gauss2_yrangem035_tdiv1qdiv50_150_10qdotscaled_50_goodduration_gamma099_hor150_lr5e-05_valuelr0.0005_bs64_constrlr0.01_nep64_neppf64_neppol32_epsppo0.05_sigmainit1.0q_1.0t_entlb-52_entlbinit52_entlbep2000_nqcps11_ntcps10_seed444/agent-444-2864.msh")
    
    #unstructured
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_unstructured_gauss10_xrangem03m06y03_qdiv100tdiv10_gamma099_hor150_lr5e-05_valuelr0.0005_bs64_constrlr0.01_nep64_neppf64_neppol32_epsppo0.05_sigmainit1.0q_1.0t_entlb-26_entlbinit52_entlbep4000_nqcps11_ntcps10_seed444/agent-444-3471.msh")

    # TRO hit
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_TROhit_cont_gauss2_yrangem035_tdiv1qdiv50_150_1_50_gamma099_hor150_lr5e-05_valuelr0.0005_bs64_constrlr0.01_nep64_neppf64_neppol32_epsppo0.05_sigmainit1.0q_1.0t_entlb-26_entlbinit52_entlbep4000_nqcps11_ntcps10_seed444/agent-444-2314.msh")

    #print("Load agent from: ", agent_path)
    #agent_ = agent_builder(env_info_, agent_params)
    #agent = Agent.load(agent_path)
    ##agent.load_robot()
    #agent._optimizer = torch.optim.Adam(agent.distribution.parameters(), lr=agent_params["mu_lr"])
    #agent.mdp_info = env_info_['rl_info']
    ##agent._epoch_no = 0
    ##agent.policy.optimizer = TrajectoryOptimizer(env_info_)
    #agent.policy.load_policy(env_info_)
    ##agent.policy.desired_ee_z = env_info_["robot"]["ee_desired_height"]
    ##agent.policy.joint_vel_limit = env_info_["robot"]["joint_vel_limit"][1] 
    ##agent.policy.joint_acc_limit = env_info_["robot"]["joint_acc_limit"][1] 
    ##agent.info.is_stateful = agent_.info.is_stateful
    ##agent.info.policy_state_shape = agent_.info.policy_state_shape
    #agent.task_losses = []
    #agent.scaled_constraint_losses = []
    #agent.task_losses = []
    ##agent.distribution._log_sigma_approximator.model.network._init_sigma *= 3.


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
        J_det, R, success, success_position, success_orientation, success_velocity, \
        states, actions, episode_length, dataset_info = compute_metrics(core, eval_params)
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
                          success=success, episode_length=episode_length)
        wandb.log({
            "Reward/": {"J_det": J_det, "J_sto": J_sto, "V_sto": V_sto, "VJ_bias": VJ_bias, "R": R, "success": success,
                        "success_position": success_position, "success_orientation": success_orientation, "success_velocity": success_velocity},
            #"Entropy/": {"E": E, "entropy_bonus": core.agent._log_entropy_bonus.exp().detach().numpy()},
            "Entropy/": {"E": E},
            #"Constraints_sto/": {"avg/": {str(i): a for i, a in enumerate(constraints_violation_sto_mean)},
            #                     "max/": {str(i): a for i, a in enumerate(constraints_violation_sto_max)}},
            #"Constraints_det/": {"avg/": {str(i): a for i, a in enumerate(constraints_violation_det_mean)},
            #                     "max/": {str(i): a for i, a in enumerate(constraints_violation_det_max)}},
            "Constraints_det_pos/": {"avg": {str(i): a for i, a in enumerate(constraints_violation_det_mean[:7])},
                                     "max": {str(i): a for i, a in enumerate(constraints_violation_det_max[:7])}},
            "Constraints_det_vel/": {"avg": {str(i): a for i, a in enumerate(constraints_violation_det_mean[7:14])},
                                     "max": {str(i): a for i, a in enumerate(constraints_violation_det_max[7:14])}},
            "Constraints_det_acc/": {"avg": {str(i): a for i, a in enumerate(constraints_violation_det_mean[14:21])},
                                     "max": {str(i): a for i, a in enumerate(constraints_violation_det_max[14:21])}},
            "Constraints_det_torque/": {"avg": {str(i): a for i, a in enumerate(constraints_violation_det_mean[21:28])},
                                        "max": {str(i): a for i, a in enumerate(constraints_violation_det_max[21:28])}},
            "Constraints_det_orientation/": {"avg": {str(i): a for i, a in enumerate(constraints_violation_det_mean[28:29])},
                                             "max": {str(i): a for i, a in enumerate(constraints_violation_det_max[28:29])}},
            "Constraints_det_collision/": {"avg": {str(i): a for i, a in enumerate(constraints_violation_det_mean[29:33])},
                                             "max": {str(i): a for i, a in enumerate(constraints_violation_det_max[29:33])}},
            "Stats/": {
                "mean_duration": mean_duration,
                "episode_length": episode_length,
            },
            "Constraints/": {
                    "joint_pos": np.mean(constraints_violation_det_mean[:7]),
                    "joint_vel": np.mean(constraints_violation_det_mean[7:14]),
                    "joint_acc": np.mean(constraints_violation_det_mean[14:21]),
                    "joint_torque": np.mean(constraints_violation_det_mean[21:28]),
                    "orientaiton": np.mean(constraints_violation_det_mean[28]),
                    "collision": np.mean(constraints_violation_det_mean[29:33]),
                    #"joint_vel": np.mean(dataset_info['joint_vel_constraint']),
                    #"joint_acc": np.mean(dataset_info['joint_acc_constraint']),
                    #"joint_torque": np.mean(dataset_info['joint_torque_constraint']),
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
    success_position = 0
    success_orientation = 0
    success_velocity = 0
    current_idx = 0
    scored = []
    for episode_len in eps_length:
        #success += dataset.info["success"][current_idx + episode_len - 1]
        #success_position += dataset.info["success_position"][current_idx + episode_len - 1]
        #success_orientation += dataset.info["success_orientation"][current_idx + episode_len - 1]
        #success_velocity += dataset.info["success_velocity"][current_idx + episode_len - 1]
        success += np.mean(dataset.info["success"][current_idx:current_idx + episode_len])
        success_position += np.mean(dataset.info["success_position"][current_idx:current_idx + episode_len - 1])
        success_orientation += np.mean(dataset.info["success_orientation"][current_idx:current_idx + episode_len - 1])
        success_velocity += np.mean(dataset.info["success_velocity"][current_idx:current_idx + episode_len - 1])
        current_idx += episode_len
    success /= len(eps_length)
    success_position /= len(eps_length)
    success_orientation /= len(eps_length)
    success_velocity /= len(eps_length)

    state = dataset.state
    action = dataset.action

    return J, R, success, success_position, success_orientation, success_velocity, state, action, np.mean(eps_length), dataset.info


if __name__ == "__main__":
    run_experiment(experiment)
