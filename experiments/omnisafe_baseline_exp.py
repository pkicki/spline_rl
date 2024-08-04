import omnisafe
from experiment_launcher import single_experiment, run_experiment

from spline_rl.envs.omnisafe_wrapper import OmnisafeWrapper

@single_experiment
def experiment(
    alg: str = 'TRPOLag',
    env: str = 'kinodynamic',
    n_episodes: int = 256,
    avg_steps_per_episode: int = 100,
    n_epochs: int = 800,
    n_eval_episodes: int = 25,
    batch_size: int = 64,
    actor_lr: float = 5e-4,
    critic_lr: float = 5e-4,
    cost_limit: float =  1e1,
    lambda_lr: float = 0.01,
    group_name_postfix: str = '',
    results_dir: str = './logs',
    seed: int = 444,
    **kwargs
):
    wandb_group = f"{alg}_{env}_cl{cost_limit}_{group_name_postfix}"
    if "Lag" in alg:
        wandb_group += f"_clr{lambda_lr}"
    custom_cfgs = {
        "train_cfgs": {
            "device": "cpu",
            "total_steps": n_episodes * avg_steps_per_episode * n_epochs,
            "vector_env_nums": 1,
            "torch_threads": 1,
        },
        "algo_cfgs": {
            "steps_per_epoch": n_episodes * avg_steps_per_episode,
            'update_iters': 32,
            "batch_size": batch_size,
            "kl_early_stop": False,
        },
        "logger_cfgs": {
            "use_wandb": False,
            #"use_wandb": True,
            "wandb_project": "omnisafe",
            "wandb_group": wandb_group,
            "use_tensorboard": False,
            "save_model_freq": 50,
        },
        "model_cfgs": {
            "linear_lr_decay": False,
            "actor": {
                "hidden_sizes": [256, 256],
                "lr": actor_lr,
            },
            "critic": {
                "hidden_sizes": [256, 256],
                "lr": critic_lr,
            },
        },
        "seed": seed,
    }

    if alg == "PCPO":
        custom_cfgs["algo_cfgs"]["cost_limit"] = cost_limit

    if "Lag" in alg:
        custom_cfgs["lagrange_cfgs"] = {
            "cost_limit": cost_limit,
            "lagrangian_multiplier_init": 1.,
            "lambda_lr": lambda_lr,
        }

    if alg == "PPOLag":
        custom_cfgs["algo_cfgs"]["clip"] = 0.05

    agent = omnisafe.Agent(
        alg,
        env,
        custom_cfgs=custom_cfgs,
    )

    a = omnisafe.utils.config.Config()

    agent.learn()
    agent.evaluate(num_episodes=n_eval_episodes)

if __name__ == "__main__":
    run_experiment(experiment)