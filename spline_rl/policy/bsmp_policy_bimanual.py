import numpy as np
import torch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from spline_rl.policy.bsmp_policy import BSMPPolicy
from spline_rl.utils.bspline import BSpline
from spline_rl.utils.utils import unpack_data_bimanual


class BSMPPolicyBimanual(BSMPPolicy):
    def __init__(self, env_info, dt, n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end,
                 t_scale=1., q_scale=1., q_d_scale=1., q_dot_d_scale=1., q_ddot_d_scale=1.):
        super().__init__(env_info, dt, n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end,
                         t_scale, q_scale, q_d_scale, q_dot_d_scale, q_ddot_d_scale)

        self.q_d_bias = torch.tensor([
            0., # base joint
            2.2220e-01, 4.9374e-01,  5.6134e-01, -1.7766e-01, -9.0099e-03,  3.1140e-01, # left arm
            -5.3729e-02, -6.5654e-01, -5.3388e-01,  2.0437e-01, -1.3195e-02, -7.5073e-02]) # right arm

        self._add_save_attr(
            q_d_bias='pickle',
        )

    def unpack_context(self, context):
        q_0, q_dot_0, left_ee_pos, right_ee_pos, plate_pos, plate_rot, plate_vel, \
        pegs_pos, pegs_rot = unpack_data_bimanual(torch.tensor(context))
        q_0 = q_0[:, None]
        q_dot_0 = q_dot_0[:, None]
        q_ddot_0 = torch.zeros_like(q_0)
        q_dot_d = torch.zeros_like(q_dot_0)
        q_ddot_d = torch.zeros_like(q_ddot_0)
        return q_0, q_dot_0, q_ddot_0, None, q_dot_d, q_ddot_d

    def compute_trajectory_from_theta(self, theta, context):
        q_0, q_dot_0, q_ddot_0, _, q_dot_d, q_ddot_d = self.unpack_context(context)
        trainable_q_cps, trainable_t_cps = self.extract_qt(theta)
        trainable_t_cps = trainable_t_cps * self.t_scale
        trainable_q_middle = trainable_q_cps[:, :-1] * self.q_scale
        trainable_q_d_ = trainable_q_cps[:, -1:] * self.q_d_scale
        trainable_q_pts = torch.tanh(trainable_q_middle) * 2. * np.pi
        trainable_q_d = torch.tanh(trainable_q_d_) * 2. * np.pi

        #q_d = trainable_q_d + q_0#q_d_bias
        q_d = trainable_q_d + self.q_d_bias[None, None]

        q1, q2, qm2, qm1 = self.compute_boundary_control_points_exp(trainable_t_cps, q_0, q_dot_0, q_ddot_0,
                                                                    q_d, q_dot_d, q_ddot_d)
        q_begin = [q_0, q1, q2]
        q_end = [q_d, qm1, qm2]

        s = torch.linspace(0., 1., trainable_q_pts.shape[1]+6)[None, 3:-3, None]
        q_b = q_0 * (1 - s) + q_d * s
        q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_b + trainable_q_pts] + q_end[::-1], axis=-2)

        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps, trainable_t_cps, differentiable=True)

        self._traj_no += 1
        return q, q_dot, q_ddot, t, dt, duration