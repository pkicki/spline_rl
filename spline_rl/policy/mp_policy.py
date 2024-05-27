import numpy as np
import torch
from scipy.interpolate import interp1d

from mushroom_rl.policy import Policy
from baseline.baseline_agent.optimizer import TrajectoryOptimizer

from mushroom_rl.features._implementations.basis_features import BasisFeatures
from mushroom_rl.features.basis import GaussianRBF
from spline_rl.utils.gaussian_derivative import dGaussianRBF
from spline_rl.utils.utils import unpack_data_airhockey


class MPPolicy(Policy):
    def __init__(self, env_info, n_q_cps, n_dim, n_pts_fixed_begin=1,
                 t_scale=1., q_scale=1., q_d_scale=1., q_dot_d_scale=1., q_ddot_d_scale=1., **kwargs):
        self.n_dim = n_dim
        self._n_q_pts = n_q_cps
        self._n_pts_fixed_begin = n_pts_fixed_begin
        self._n_trainable_q_pts = self._n_q_pts - self._n_pts_fixed_begin
        self.t_scale = t_scale
        self.q_scale = q_scale
        self.q_d_scale = q_d_scale
        self.q_dot_d_scale = q_dot_d_scale
        self.q_ddot_d_scale = q_ddot_d_scale

        self._traj_no = 0

        self.desired_ee_z = env_info['robot']['ee_desired_height']
        self.joint_vel_limit = env_info['robot']['joint_vel_limit'][1]
        self.joint_acc_limit = env_info['robot']['joint_acc_limit'][1]

        self.dt = env_info['dt']
        self.horizon = env_info['rl_info'].horizon


        self.load_policy(env_info)
        self.generate_basis()

        policy_state_shape = (1,)
        super().__init__(policy_state_shape)

        self._add_save_attr(
            dt='primitive',
            horizon='primitive',
            n_dim='primitive',
            _n_q_pts='primitive',
            _n_pts_fixed_begin='primitive',
            _n_trainable_q_pts='primitive',
            _traj_no='primitive',
            desired_ee_z='primitive',
            joint_vel_limit='pickle',
            joint_acc_limit='pickle',
        )

    def generate_basis(self):
        pass
    
    def load_policy(self, env_info):
        self.optimizer = TrajectoryOptimizer(env_info)

    def unpack_context(self, context):
        if context is None:
            raise NotImplementedError
        else:
            puck, puck_dot, q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, opponent_mallet = unpack_data_airhockey(torch.tensor(context))
        return q_0[:, None], q_d[:, None], q_dot_0[:, None], q_dot_d[:, None], q_ddot_0[:, None], q_ddot_d[:, None], puck

    def compute_trajectory_from_theta(self, theta, context):
        raise NotImplementedError


    def reset(self, initial_state=None):
        if initial_state is None:
            return None
        else:
            if len(initial_state.shape) == 1:
                initial_state = initial_state[None]
            q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory_from_theta(self._weights, initial_state)
            q = q.detach().numpy()
            q_dot = q_dot.detach().numpy()
            q_ddot = q_ddot.detach().numpy()
            t = t.detach().numpy()
            #duration = duration.detach().numpy()
            self.q = interp1d(t[0], q[0], axis=0)
            self.q_dot = interp1d(t[0], q_dot[0], axis=0)
            self.q_ddot = interp1d(t[0], q_ddot[0], axis=0)
            self.duration = duration[0]
            return torch.tensor([0], dtype=torch.int32)
        

    def draw_action(self, state, policy_state=None):
        """
        Args:
            state (ndarray): state of the system
            policy_state (ndarray, None): the policy internal state.

        Returns:
            numpy.ndarray, (3, num_joints): The desired [Positions, Velocities, Acceleration] of the
            next step. The environment will take first two arguments of the to control the robot.
            The third array is used for the training of the SAC as the output is acceleration. This
            action tuple will be saved in the dataset buffer
        """
        assert policy_state is not None
        t = policy_state[0] * self.dt
        if t <= self.duration:
            q = self.q(t)
            q_dot = self.q_dot(t)
            q_ddot = self.q_ddot(t)
        else:
            q = self.q(self.duration)
            q_dot = np.zeros_like(q)
            q_ddot = np.zeros_like(q)
        policy_state[0] += 1
        #q_dot = np.clip(q_dot, -self.joint_vel_limit, self.joint_vel_limit)
        #action = np.stack([q, q_dot], axis=-2) 
        action = np.stack([q, q_dot, q_ddot], axis=-2) 
        action = torch.tensor(action, dtype=torch.float32)
        return action, torch.tensor(policy_state)

    def set_weights(self, weights):
        self._weights = weights

    def compute_trajectory(self, q_cps, t_scale, differentiable=False):
        raise NotImplementedError