from itertools import product

from experiment_launcher import Launcher, is_local
import os 

os.environ["WANDB_API_KEY"] = "a9819ac569197dbd24b580d854c3041ad75efafd"

LOCAL = is_local()
TEST = False
USE_CUDA = False

N_SEEDS = 5
if LOCAL:
    N_EXPS_IN_PARALLEL = 4
else:
    N_EXPS_IN_PARALLEL = 80

N_CORES = 1
MEMORY_SINGLE_JOB = 2000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'standard'  # 'amd', 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
#CONDA_ENV = None
CONDA_ENV = "saferl"

#experiment_name = 'bsmp_eppo_kinodynamic'
#experiment_name = 'bsmp_eppo_box_pushing_fixed_notcpenergyloss_explorationswipe'
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

#for q_scale in [1./50., 1./20, 1./10.]:
#    launcher.add_experiment(
#        #alg="bsmp_eppo_stop",
#        #alg="bsmp_eppo_kinodynamic",
#        alg="bsmp_eppo_box_pushing",
#        group_name=experiment_name,
#
#        q_scale__ = q_scale,
#        q_d_scale__ = q_scale,
#
#        ## kinodynamic
#        #initial_entropy_lb = 45,
#        #entropy_lb = -45 / 2.,
#        #q_d_scale = 1. / 150., # structured
#        ##q_d_scale = 1. / 50., # unstructured
#
#
#        # air hockey
#        #reward_type="puze", # available options are "new", "puze", "mixed"
#        #mode="disabled",
#    )
#launcher.run(LOCAL, TEST)

#mu_lrs = [1e-4, 3e-5, 1e-5]
#value_lrs = [1e-3, 3e-4, 1e-4]
#constraint_lrs = [1e-2, 3e-3, 1e-3]

t_scales = [1., 0.1]
q_scales = [1./3000., 1./5000.]
q_d_scales = [1./1500., 1./3000.]
kl_thresholds = [0.1, 1e10]

#for mu_lr, value_lr, constraint_lr in product(mu_lrs, value_lrs, constraint_lrs):
for t_scale, q_scale, q_d_scale, kl_threshold in product(t_scales, q_scales, q_d_scales, kl_thresholds):
    launcher.add_experiment(
        #alg="bsmp_eppo_stop",
        #alg="bsmp_eppo_kinodynamic",
        #alg="bsmp_eppo_box_pushing",
        alg="bsmp_eppo_bimanual",
        #group_name=experiment_name,
        group_name=f"{experiment_name}_t{t_scale}q{q_scale}qd{q_d_scale}_klth{kl_threshold}",

        # bimanual
        #constraint_lr__=constraint_lr,
        #mu_lr__=mu_lr,
        #value_lr__=value_lr,
        t_scale__=t_scale,
        q_scale__=q_scale,
        q_d_scale__=q_d_scale,
        kl_threshold__=kl_threshold,

        ## kinodynamic
        #initial_entropy_lb = 45,
        #entropy_lb = -45 / 2.,
        #q_d_scale = 1. / 150., # structured
        ##q_d_scale = 1. / 50., # unstructured


        # air hockey
        #reward_type="puze", # available options are "new", "puze", "mixed"
        #mode="disabled",
    )
launcher.run(LOCAL, TEST)