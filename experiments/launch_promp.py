import os
from itertools import product
from experiment_launcher import Launcher, is_local

os.environ["WANDB_API_KEY"] = "a9819ac569197dbd24b580d854c3041ad75efafd"

LOCAL = is_local()
TEST = False
USE_CUDA = False

N_SEEDS = 2
if LOCAL:
    N_EXPS_IN_PARALLEL = 1
else:
    N_EXPS_IN_PARALLEL = 15

N_CORES = 1
MEMORY_SINGLE_JOB = 2000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'amd2,amd'  # 'amd', 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = "saferl"

experiment_name = 'promp_eppo_bimanual'

launcher = Launcher(
    exp_name=experiment_name,
    #exp_file='air_hockey_episodic_exp',
    exp_file='bimanual_episodic_exp',
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
    alg="promp_eppo_bimanual",
    group_name=f"medium_{experiment_name}_t01q1500qd1500_klth1e10_ss100",

    initial_entropy_lb=131.,
    entropy_lb=-131. / 2.,

    n_episodes=256,
    n_episodes_per_fit = 64,
    n_eval_episodes = 25,
    batch_size= 64,
    #n_epochs=2,
    #n_episodes=4,
    #n_episodes_per_fit=4,
    #n_eval_episodes=2,
    #batch_size=4,
    #debug=True,

    # bimanual
    t_scale=0.1,
    q_scale=1./1500.,
    q_d_scale=1./1500.,
    kl_threshold=1e10,
    #kl_threshold=2e-2,
    #cov="full",
    value_function_bias=10.,
    success_scale=100.,
)
launcher.run(LOCAL, TEST)