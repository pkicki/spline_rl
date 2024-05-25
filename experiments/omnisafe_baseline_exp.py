import sys
import omnisafe
from spline_rl.envs.omnisafe_wrapper import OmnisafeWrapper

n_episodes = 256
#n_episodes = 4
avg_steps_per_episode = 65
#n_epochs = 5
n_epochs = 5000
n_eval_episodes = 25
batch_size = 64
actor_lr = 5e-4
critic_lr = 5e-4
cost_limit =  1e-3
lambda_lr = 0.01
seed = sys.argv[1] if len(sys.argv) > 1 else 444

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
        "clip": 0.05,
    },
    "logger_cfgs": {
        "use_wandb": True,
        "wandb_project": "omnisafe",
        "wandb_group": "PPOLag_cl1em3",
        "use_tensorboard": False,
        "save_model_freq": 100,
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
    'PPOLag',
    'air_hockey',
    custom_cfgs=custom_cfgs,
)

a = omnisafe.utils.config.Config()

agent.learn()
agent.evaluate(num_episodes=n_eval_episodes)
#agent.render(num_episodes=1)