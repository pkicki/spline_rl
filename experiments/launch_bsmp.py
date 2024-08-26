from itertools import product

from experiment_launcher import Launcher, is_local

LOCAL = is_local()
TEST = False
USE_CUDA = False

N_SEEDS = 1
if LOCAL:
    N_EXPS_IN_PARALLEL = 2
else:
    N_EXPS_IN_PARALLEL = 10

N_CORES = 1
MEMORY_SINGLE_JOB = 2000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'standard'  # 'amd', 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = None

#experiment_name = 'bsmp_eppo_kinodynamic'
#experiment_name = 'bsmp_eppo_box_pushing'
experiment_name = 'bsmp_eppo_bimanual'

launcher = Launcher(
    exp_name=experiment_name,
    #exp_file='air_hockey_episodic_exp',
    #exp_file='kinodynamic_cup_episodic_exp',
    #exp_file='box_pushing_episodic_exp',
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

mu_lrs = [1e-4, 3e-5, 1e-5]
value_lrs = [1e-3, 3e-4, 1e-4]
constraint_lrs = [1e-2, 3e-3, 1e-3]

for mu_lr, value_lr, constraint_lr in product(mu_lrs, value_lrs, constraint_lrs):
    launcher.add_experiment(
        #alg="bsmp_eppo_stop",
        #alg="bsmp_eppo_kinodynamic",
        #alg="bsmp_eppo_box_pushing",
        alg="bsmp_eppo_bimanual",
        group_name=experiment_name,

        # bimanual
        constraint_lr__=constraint_lr,
        mu_lr__=mu_lr,
        value_lr__=value_lr,
        t_scale=1.0,
        q_scale=1./1500.,
        q_d_scale=1./500.,

        ## kinodynamic
        #initial_entropy_lb = 45,
        #entropy_lb = -45 / 2.,
        #q_d_scale = 1. / 150., # structured
        ##q_d_scale = 1. / 50., # unstructured


        # air hockey
        #reward_type="puze", # available options are "new", "puze", "mixed"
        mode="disabled",
    )
launcher.run(LOCAL, TEST)