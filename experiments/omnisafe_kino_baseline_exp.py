import sys
import omnisafe
from experiment_launcher import single_experiment, run_experiment

from spline_rl.envs.omnisafe_wrapper import OmnisafeWrapper

@single_experiment
def experiment(
    alg: str = 'TRPOLag',
    n_episodes: int = 256,
    avg_steps_per_episode: int = 100,
    n_epochs: int = 800,
    n_eval_episodes: int = 25,
    batch_size: int = 64,
    actor_lr: float = 5e-4,
    critic_lr: float = 5e-4,
    cost_limit: float =  1e1,
    lambda_lr: float = 0.01,
    results_dir: str = './logs',
    seed: int = 444,
    **kwargs
):
    #seed = sys.argv[1] if len(sys.argv) > 1 else 444

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
            # PPOLag
            #"clip": 0.05,
        },
        "logger_cfgs": {
            "use_wandb": False,
            #"use_wandb": True,
            "wandb_project": "omnisafe",
            #"wandb_group": "PPOLag_kino_cl1e1_clr1em2",
            "wandb_group": f"{alg}_kino_cl1e1_clr1em2",
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
        "lagrange_cfgs": {
            "cost_limit": cost_limit,
            "lagrangian_multiplier_init": 1.,
            "lambda_lr": lambda_lr,
        },
        "seed": seed,
    }
    agent = omnisafe.Agent(
        #'PPOLag',
        #'TRPOLag',
        alg,
        'kinodynamic',
        custom_cfgs=custom_cfgs,
    )

    a = omnisafe.utils.config.Config()

    agent.learn()
    #agent.evaluate(num_episodes=n_eval_episodes)
    #agent.render(num_episodes=1)

if __name__ == "__main__":
    run_experiment(experiment)