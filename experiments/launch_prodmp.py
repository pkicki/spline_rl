from itertools import product

from experiment_launcher import Launcher, is_local

LOCAL = is_local()
TEST = False
USE_CUDA = False

N_SEEDS = 1
if LOCAL:
    N_EXPS_IN_PARALLEL = 1
else:
    N_EXPS_IN_PARALLEL = 10

N_CORES = 4
MEMORY_SINGLE_JOB = 1000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'amd2,amd'  # 'amd', 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = None

experiment_name = 'prodmp_eppo_unstructured'

launcher = Launcher(
    exp_name=experiment_name,
    exp_file='air_hockey_episodic_exp',
    # project_name='project01234',  # for hrz cluster
    n_seeds=N_SEEDS,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=3,
    hours=23,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=True,
    compact_dirs=False
)

for reward_type in ["new"]:
    launcher.add_experiment(
        alg="prodmp_eppo_unstructured",
        reward_type__=reward_type,
        group_name__=f"{experiment_name}_{reward_type}",
        n_epochs=2000,
        n_episodes=256,
        n_episodes_per_fit = 64,
        n_eval_episodes = 25,
        batch_size= 64,
        #n_episodes=4,
        #n_episodes_per_fit = 4,
        #n_eval_episodes = 2,
        #batch_size= 4,
        mode="disabled",
    )
launcher.run(LOCAL, TEST)