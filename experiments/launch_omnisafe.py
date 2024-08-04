import os
from itertools import product
from experiment_launcher import Launcher, is_local

os.environ['WANDB_API_KEY'] = "a9819ac569197dbd24b580d854c3041ad75efafd"

LOCAL = is_local()
TEST = False
USE_CUDA = False

N_SEEDS = 2
if LOCAL:
    N_EXPS_IN_PARALLEL = 2
else:
    N_EXPS_IN_PARALLEL = 10

N_CORES = 1
MEMORY_SINGLE_JOB = 2000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'standard'  # 'amd', 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = "saferl"

#env = 'kinodynamic'
env = 'air_hockey'

if env == 'kinodynamic':
    n_epochs = 800
    avg_steps_per_episode = 100
    cost_limit = 1e1
    lambda_lr = 1e-2 
elif env == 'air_hockey':
    n_epochs = 3000
    avg_steps_per_episode = 65
    cost_limit = 1e-3
    lambda_lr = 1e-2

alg = 'PPOLag'
#alg = 'TRPOLag'
#alg = 'PCPO'

experiment_name = f'omnisafe_{env}_{alg}'

launcher = Launcher(
    exp_name=experiment_name,
    exp_file='omnisafe_baseline_exp',
    # project_name='project01234',  # for hrz cluster
    n_seeds=N_SEEDS,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=6,
    hours=23,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=True,
    compact_dirs=False
)

launcher.add_experiment(
    alg=alg,
    env=env,
    group_name=experiment_name,
    avg_steps_per_episode=avg_steps_per_episode,
    cost_limit=cost_limit,
    lambda_lr=lambda_lr,
    #mode="disabled",
)
launcher.run(LOCAL, TEST)