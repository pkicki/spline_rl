from itertools import product

from experiment_launcher import Launcher, is_local

LOCAL = is_local()
TEST = False
USE_CUDA = False

N_SEEDS = 2
if LOCAL:
    N_EXPS_IN_PARALLEL = 1
else:
    N_EXPS_IN_PARALLEL = 1

N_CORES = 1
MEMORY_SINGLE_JOB = 1000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'fast'  # 'amd', 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = None

experiment_name = 'bsmp_eppo_structured'

launcher = Launcher(
    exp_name=experiment_name,
    exp_file='air_hockey_episodic_exp',
    # project_name='project01234',  # for hrz cluster
    n_seeds=N_SEEDS,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=0,
    hours=0,
    minutes=1,
    seconds=0,
    partition=PARTITION,
    gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=True,
    compact_dirs=False
)

for reward_type in ["puze"]:
    launcher.add_experiment(
        alg="bsmp_eppo_stop",
        reward_type__=reward_type,
        group_name__=f"{experiment_name}_{reward_type}",
        n_epochs=5,
        n_episodes=2,
        n_episodes_per_fit = 2,
        n_eval_episodes = 2,
        batch_size= 2,
        #mode="disabled",
    )
launcher.run(LOCAL, TEST)