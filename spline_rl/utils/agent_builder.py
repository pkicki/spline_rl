import torch

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator

from spline_rl.distribution.bsmp_distribution import DiagonalGaussianBSMPSigmaDistribution
from spline_rl.policy.bsmp_policy import BSMPPolicy
from spline_rl.policy.bsmp_unstructured_policy import BSMPUnstructuredPolicy
from spline_rl.policy.prodmp_policy import ProDMPPolicy
from spline_rl.policy.promp_policy import ProMPPolicy
from spline_rl.policy.bsmp_policy_bimanual import BSMPPolicyBimanual
from spline_rl.policy.bsmp_policy_box_pushing import BSMPPolicyBoxPushing
from spline_rl.policy.bsmp_policy_kino import BSMPPolicyKino
from spline_rl.policy.bsmp_unstructured_policy_kino import BSMPPolicyUnstructuredKino
from spline_rl.policy.prodmp_policy_kino import ProDMPPolicyKino
from spline_rl.policy.promp_policy_kino import ProMPPolicyKino
from spline_rl.policy.bsmp_policy_stop import BSMPPolicyStop
from spline_rl.algorithm.bsmp_eppo import BSMPePPO
from spline_rl.utils.context_builder import IdentityContextBuilder
from spline_rl.utils.basic_network import BasicConfigurationNetworkWrapper, BasicConfigurationTimeNetworkWrapper, BasicLogSigmaNetworkWrapper
from spline_rl.utils.air_hockey_network import AirHockeyConfigurationNetworkWrapper, AirHockeyConfigurationTimeNetworkWrapper, AirHockeyLogSigmaNetworkWrapper
from spline_rl.utils.value_network import AirHockeyValueNetwork, BasicValueNetwork



def agent_builder(env_info, agent_params):
    alg = agent_params["alg"]

    agent_params["optimizer"] = {
        'class': torch.optim.Adam,
        'params': {'lr': agent_params["mu_lr"],
                   'weight_decay': 0.0}}

    eppo_params = dict(n_epochs_policy=agent_params["n_epochs_policy"],
                       batch_size=agent_params["batch_size"],
                       eps_ppo=agent_params["eps_ppo"],
                       kl_threshold=agent_params["kl_threshold"],
                       context_builder=IdentityContextBuilder(),
                       )

    if alg.startswith("bsmp"):
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
    t_scale = agent_params["t_scale"]
    q_scale = agent_params["q_scale"]
    q_d_scale = agent_params["q_d_scale"]
    q_dot_d_scale = agent_params["q_dot_d_scale"]
    q_ddot_d_scale = agent_params["q_ddot_d_scale"]

    if "stop" in agent_params["alg"]:
        n_trainable_q_pts += n_q_pts - 5
        n_trainable_t_pts += n_t_pts


    mdp_info = env_info['rl_info']

    sigma_q = agent_params["sigma_init_q"] * torch.ones((n_trainable_q_pts, n_dim))
    sigma_t = agent_params["sigma_init_t"] * torch.ones((n_trainable_t_pts))
    sigma = torch.cat([sigma_q.reshape(-1), sigma_t]).type(torch.FloatTensor)

    #if "kinodynamic" in agent_params["alg"]:
    #    mu_network = KinoConfigurationTimeNetworkWrapper
    #    logsigma_network = KinoLogSigmaNetworkWrapper
    #    value_netwotk = KinoValueNetwork
    #elif "box_pushing" in agent_params["alg"]:
    #    mu_network = BoxPushingConfigurationTimeNetworkWrapper
    #    logsigma_network = BoxPushingLogSigmaNetworkWrapper
    #    value_netwotk = BoxPushingValueNetwork
    #elif "bimanual" in agent_params["alg"]:
    #    mu_network = BimanualConfigurationTimeNetworkWrapper
    #    logsigma_network = BimanualLogSigmaNetworkWrapper
    #    value_netwotk = BimanualValueNetwork
    #else:
    #    mu_network = ConfigurationTimeNetworkWrapper
    #    logsigma_network = LogSigmaNetworkWrapper
    #    value_netwotk = ValueNetwork

    if "air_hockey" in agent_params["env"]:
        mu_network = AirHockeyConfigurationTimeNetworkWrapper
        logsigma_network = AirHockeyLogSigmaNetworkWrapper
        value_network = AirHockeyValueNetwork
    else:
        mu_network = BasicConfigurationTimeNetworkWrapper
        logsigma_network = BasicLogSigmaNetworkWrapper
        value_network = BasicValueNetwork
    

    mu_approximator = Regressor(TorchApproximator,
                                network=mu_network,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_dim * n_trainable_q_pts, n_trainable_t_pts))
    log_sigma_approximator = Regressor(TorchApproximator,
                                network=logsigma_network,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        "init_sigma": sigma,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_dim * n_trainable_q_pts + n_trainable_t_pts,))

    value_function_approximator = value_network(mdp_info.observation_space, agent_params["value_function_bias"])

    policy_args = dict(
        env_info=env_info,
        dt=env_info["dt"],
        n_q_pts=n_q_pts,
        n_dim=n_dim,
        n_t_pts=n_t_pts,
        n_pts_fixed_begin=n_pts_fixed_begin,
        n_pts_fixed_end=n_pts_fixed_end,
        t_scale=t_scale,
        q_scale=q_scale,
        q_d_scale=q_d_scale,
        q_dot_d_scale=q_dot_d_scale,
        q_ddot_d_scale=q_ddot_d_scale,
    )

    if agent_params["alg"] == "bsmp_eppo_unstructured":
        policy = BSMPUnstructuredPolicy(**policy_args)
    elif agent_params["alg"] == "bsmp_eppo_stop":
        policy = BSMPPolicyStop(**policy_args)
    elif agent_params["alg"] == "bsmp_eppo_kinodynamic":
        policy = BSMPPolicyKino(**policy_args)
    elif agent_params["alg"] == "bsmp_eppo_kinodynamic_unstructured":
        policy = BSMPPolicyUnstructuredKino(**policy_args)
    elif agent_params["alg"] == "bsmp_eppo_box_pushing":
        policy = BSMPPolicyBoxPushing(**policy_args)
    elif agent_params["alg"] == "bsmp_eppo_bimanual":
        policy = BSMPPolicyBimanual(**policy_args)
    else:
        policy = BSMPPolicy(**policy_args)

    dist = DiagonalGaussianBSMPSigmaDistribution(mu_approximator, log_sigma_approximator, agent_params["entropy_lb"])

    value_function_optimizer = torch.optim.Adam(value_function_approximator.parameters(), lr=agent_params["value_lr"])

    agent = BSMPePPO(mdp_info, dist, policy, agent_params["optimizer"], value_function_approximator, value_function_optimizer,
                     agent_params["constraint_lr"], **eppo_params)
    return agent


def build_agent_ProMPePPO(env_info, eppo_params, agent_params):
    n_trainable_pts = agent_params["n_dim"] * (agent_params["n_q_cps"] - 1)
    mdp_info = env_info['rl_info']

    sigma = agent_params["sigma_init_q"] * torch.ones(n_trainable_pts + 1)

    if "air_hockey" in agent_params["env"]:
        mu_network = AirHockeyConfigurationNetworkWrapper
        logsigma_network = AirHockeyLogSigmaNetworkWrapper
        value_network = AirHockeyValueNetwork
    else:
        mu_network = BasicConfigurationNetworkWrapper
        logsigma_network = BasicLogSigmaNetworkWrapper
        value_network = BasicValueNetwork

    mu_approximator = Regressor(TorchApproximator,
                                network=mu_network,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_trainable_pts + 1,))
    log_sigma_approximator = Regressor(TorchApproximator,
                                network=logsigma_network,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        "init_sigma": sigma,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_trainable_pts + 1,))

    value_function_approximator = value_network(mdp_info.observation_space, agent_params["value_function_bias"])


    if agent_params["alg"] == "promp_eppo_unstructured":
        policy = ProMPPolicy(env_info, **agent_params)
    elif agent_params["alg"] == "promp_eppo_kinodynamic":
        policy = ProMPPolicyKino(env_info, **agent_params)
    elif agent_params["alg"] == "prodmp_eppo_unstructured":
         policy = ProDMPPolicy(env_info, **agent_params)
    elif agent_params["alg"] == "prodmp_eppo_kinodynamic":
         policy = ProDMPPolicyKino(env_info, **agent_params)
    else:
        raise ValueError(f"Unknown algorithm: {agent_params['alg']}")

    dist = DiagonalGaussianBSMPSigmaDistribution(mu_approximator, log_sigma_approximator, agent_params["entropy_lb"])

    value_function_optimizer = torch.optim.Adam(value_function_approximator.parameters(), lr=agent_params["value_lr"])

    agent = BSMPePPO(mdp_info, dist, policy, agent_params["optimizer"], value_function_approximator, value_function_optimizer,
                     agent_params["constraint_lr"], **eppo_params)
    return agent
