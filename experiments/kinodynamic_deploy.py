import os
import torch.random

from mushroom_rl.core.agent import Agent
from spline_rl.utils.env_builder import env_builder

torch.set_default_dtype(torch.float32)

def load_env_agent(agent_path, q_scale=1./50., q_d_scale=1./150.):
    env_params = dict(
        horizon=100,
        gamma=0.99,
        interpolation_order=3 if "prodmp" in agent_path else 5,
    )

    env, env_info_ = env_builder("kinodynamic_cup", 1, env_params)

    print("Load agent from: ", agent_path)
    agent = Agent.load(agent_path)
    agent.mdp_info = env_info_['rl_info']
    agent.policy.load_policy(env_info_)
    agent.distribution._log_sigma_approximator.model.network = agent.distribution._log_sigma_approximator.model.network.to(torch.float32)
    agent.distribution._mu_approximator.model.network = agent.distribution._mu_approximator.model.network.to(torch.float32)
    agent.policy.t_scale = 1.
    agent.policy.q_scale = q_scale
    agent.policy.q_d_scale = q_d_scale
    agent.policy.q_dot_d_scale = 1. / 50.
    agent.policy.q_ddot_d_scale = 1.
    agent.policy._traj_no = 0

    if "promp" in agent_path or "prodmp" in agent_path:
        agent.policy.generate_basis()

    return env, agent

    
if __name__ == "__main__":
    agent_path = os.path.join(os.path.dirname(__file__), "trained_models/kinodynamic/ours_biased/agent-0-252.msh")
    env, agent = load_env_agent(agent_path)
    state, episode_info = env.reset(None)
    _policy_state, _current_theta = agent.episode_start(state, episode_info)
    action, policy_next_state = agent.draw_action(state, _policy_state)
    next_state, reward, absorbing, step_info = env.step(action)
    a = 0
