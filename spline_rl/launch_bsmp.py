from itertools import product

from experiment_launcher import Launcher, is_local

LOCAL = is_local()
TEST = False
USE_CUDA = False

N_SEEDS = 2
if LOCAL:
    N_EXPS_IN_PARALLEL = 2
else:
    N_EXPS_IN_PARALLEL = 10

N_CORES = 4
MEMORY_SINGLE_JOB = 1000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'amd2,amd'  # 'amd', 'rtx'
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

launcher.add_experiment(
    alg="bsmp_eppo_stop",
    group_name=experiment_name,
    reward_type="new", # available options are "new", "puze", "mixed"
)
launcher.run(LOCAL, TEST)