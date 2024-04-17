import torch

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator

from distribution.bsmp_distribution import DiagonalGaussianBSMPSigmaDistribution
from policy.bsmp_policy import BSMPPolicy
from policy.bsmp_unstructured_policy import BSMPUnstructuredPolicy
from policy.prodmp_policy import ProDMPPolicy
from policy.promp_policy import ProMPPolicy
from algorithm.bsmp_eppo import BSMPePPO
from spline_rl.policy.bsmp_policy_stop import BSMPPolicyStop
from utils.context_builder import IdentityContextBuilder
from utils.network import ConfigurationNetworkWrapper, ConfigurationTimeNetworkWrapper, LogSigmaNetworkWrapper
from utils.value_network import ValueNetwork



def agent_builder(env_info, agent_params):
    alg = agent_params["alg"]

    agent_params["optimizer"] = {
        'class': torch.optim.Adam,
        'params': {'lr': agent_params["mu_lr"],
                   'weight_decay': 0.0}}

    eppo_params = dict(n_epochs_policy=agent_params["n_epochs_policy"],
                       batch_size=agent_params["batch_size"],
                       eps_ppo=agent_params["eps_ppo"],
                       target_entropy=agent_params["target_entropy"],
                       entropy_lr=agent_params["entropy_lr"],
                       initial_entropy_bonus=agent_params["initial_entropy_bonus"],
                       context_builder=IdentityContextBuilder(),
                       )

    if alg == "bsmp_eppo":
        agent = build_agent_BSMPePPO(env_info, eppo_params, agent_params)
    elif alg == "bsmp_eppo_unstructured":
        agent = build_agent_BSMPePPO(env_info, eppo_params, agent_params)
    elif alg == "bsmp_eppo_stop":
        agent = build_agent_BSMPePPO(env_info, eppo_params, agent_params)
    elif alg.startswith("pro"):
        agent = build_agent_ProMPePPO(env_info, eppo_params, agent_params)
    else:
        raise ValueError(f"Unknown algorithm: {alg}")
    return agent

def build_agent_BSMPePPO(env_info, eppo_params, agent_params):
    n_q_pts = agent_params["n_q_cps"]
    n_t_pts = agent_params["n_t_cps"]
    n_pts_fixed_begin = agent_params["n_pts_fixed_begin"]
    n_pts_fixed_end = agent_params["n_pts_fixed_end"]
    n_dim = agent_params["n_dim"]
    n_trainable_q_pts = n_q_pts - (n_pts_fixed_begin + n_pts_fixed_end)
    n_trainable_t_pts = n_t_pts

    if "stop" in agent_params["alg"]:
        n_trainable_q_pts += n_q_pts - 5
        n_trainable_t_pts += n_t_pts


    mdp_info = env_info['rl_info']

    sigma_q = agent_params["sigma_init_q"] * torch.ones((n_trainable_q_pts, n_dim))
    sigma_t = agent_params["sigma_init_t"] * torch.ones((n_trainable_t_pts))
    sigma = torch.cat([sigma_q.reshape(-1), sigma_t]).type(torch.FloatTensor)

    mu_approximator = Regressor(TorchApproximator,
                                network=ConfigurationTimeNetworkWrapper,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_dim * n_trainable_q_pts, n_trainable_t_pts))
    log_sigma_approximator = Regressor(TorchApproximator,
                                network=LogSigmaNetworkWrapper,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        "init_sigma": sigma,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_dim * n_trainable_q_pts + n_trainable_t_pts,))

    value_function_approximator = ValueNetwork(mdp_info.observation_space)

    if agent_params["alg"] == "bsmp_eppo_unstructured":
        policy = BSMPUnstructuredPolicy(env_info, env_info["dt"], n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end)
    elif agent_params["alg"] == "bsmp_eppo_stop":
        policy = BSMPPolicyStop(env_info, env_info["dt"], n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end)
    else:
        policy = BSMPPolicy(env_info, env_info["dt"], n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end)

    dist = DiagonalGaussianBSMPSigmaDistribution(mu_approximator, log_sigma_approximator, agent_params["entropy_lb"])

    value_function_optimizer = torch.optim.Adam(value_function_approximator.parameters(), lr=agent_params["value_lr"])

    agent = BSMPePPO(mdp_info, dist, policy, agent_params["optimizer"], value_function_approximator, value_function_optimizer,
                     agent_params["constraint_lr"], **eppo_params)
    return agent


def build_agent_ProMPePPO(env_info, eppo_params, agent_params):
    n_trainable_pts = agent_params["n_dim"] * (agent_params["n_q_cps"] - 1)
    mdp_info = env_info['rl_info']

    sigma = agent_params["sigma_init_q"] * torch.ones(n_trainable_pts)

    mu_approximator = Regressor(TorchApproximator,
                                network=ConfigurationNetworkWrapper,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_trainable_pts,))
    log_sigma_approximator = Regressor(TorchApproximator,
                                network=LogSigmaNetworkWrapper,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        "init_sigma": sigma,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_trainable_pts,))

    value_function_approximator = ValueNetwork(mdp_info.observation_space)


    if agent_params["alg"] == "promp_eppo":
        policy = ProMPPolicy(env_info, agent_params["n_q_cps"], agent_params["n_dim"])
    elif agent_params["alg"] == "prodmp_eppo":
        policy = ProDMPPolicy(env_info, agent_params["n_q_cps"], agent_params["n_dim"])
    else:
        raise ValueError(f"Unknown algorithm: {agent_params['alg']}")

    dist = DiagonalGaussianBSMPSigmaDistribution(mu_approximator, log_sigma_approximator, agent_params["entropy_lb"])

    value_function_optimizer = torch.optim.Adam(value_function_approximator.parameters(), lr=agent_params["value_lr"])

    agent = BSMPePPO(mdp_info, dist, policy, agent_params["optimizer"], value_function_approximator, value_function_optimizer,
                     agent_params["constraint_lr"], **eppo_params)
    return agent