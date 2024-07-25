import os
import torch.random
import numpy as np

from mushroom_rl.core.agent import Agent
from spline_rl.utils.env_builder import env_builder

torch.set_default_dtype(torch.float32)

def load_env_agent(agent_path, interpolation_order=-1, dtype=torch.double):
    env_params = dict(
        moving_init=True,
        horizon=150,
        gamma=0.99,
        interpolation_order=interpolation_order,
        reward_type="puze",
    )

    env, env_info_ = env_builder("air_hockey", 1, env_params)

    if dtype == torch.float32:
        np_dtype = np.float32
        torch_dtype = torch.float32
    elif dtype == torch.double or dtype == torch.float64:
        np_dtype = np.float64
        torch_dtype = torch.float64

    print("Load agent from: ", agent_path)
    agent = Agent.load(agent_path)
    agent.mdp_info = env_info_['rl_info']
    agent.policy.load_policy(env_info_)
    if hasattr(agent.policy, "_q_bsp") and hasattr(agent.policy, "_t_bsp"):
        agent.policy._q_bsp.N = agent.policy._q_bsp.N.astype(np_dtype)
        agent.policy._q_bsp.dN = agent.policy._q_bsp.dN.astype(np_dtype)
        agent.policy._q_bsp.ddN = agent.policy._q_bsp.ddN.astype(np_dtype)
        agent.policy._t_bsp.N = agent.policy._t_bsp.N.astype(np_dtype)
        agent.policy._t_bsp.dN = agent.policy._t_bsp.dN.astype(np_dtype)
    agent.distribution._log_sigma_approximator.model.network = agent.distribution._log_sigma_approximator.model.network.to(torch_dtype)
    agent.distribution._mu_approximator.model.network = agent.distribution._mu_approximator.model.network.to(torch_dtype)
    agent.policy.t_scale = 1.
    agent.policy.q_scale = 1. / 50.
    agent.policy.q_d_scale = 1. / 150.
    agent.policy.q_dot_d_scale = 1. / 50.
    agent.policy.q_ddot_d_scale = 1.
    agent.policy._traj_no = 0

    return env, agent

    
if __name__ == "__main__":
    agent_path = os.path.join(os.path.dirname(__file__), "trained_models/bsmp_eppo_stop_puzereward/agent-7-1382.msh")
    env, agent = load_env_agent(agent_path)
    state, episode_info = env.reset(None)
    _policy_state, _current_theta = agent.episode_start(state, episode_info)
    action, policy_next_state = agent.draw_action(state, _policy_state)
    next_state, reward, absorbing, step_info = env.step(action)
    a = 0
