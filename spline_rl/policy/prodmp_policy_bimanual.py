import numpy as np
from spline_rl.policy.prodmp_policy import ProDMPPolicy
import torch

from spline_rl.utils.utils import unpack_data_bimanual

import matplotlib.pyplot as plt


class ProDMPPolicyBimanual(ProDMPPolicy):
    def __init__(self, env_info, n_q_cps, n_dim, n_pts_fixed_begin=1, t_scale=1, q_scale=1, q_d_scale=1, q_dot_d_scale=1, q_ddot_d_scale=1, **kwargs):
        super().__init__(env_info, n_q_cps, n_dim, n_pts_fixed_begin, t_scale, q_scale, q_d_scale, q_dot_d_scale, q_ddot_d_scale, **kwargs)
        self.q_d_bias = torch.tensor([
            0.,
            0.21646051,  0.48360129,  0.55410618, -0.17025892, -0.00847046,  0.30138272,
            -0.04727616, -0.64639233, -0.52647715, 0.1963449 , -0.01348486, -0.06694195])

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
        theta = theta.to(torch.float64) if type(theta) is torch.Tensor else theta.astype(np.float64)
        q_0, q_dot_0, q_ddot_0, _, q_dot_d, q_ddot_d = self.unpack_context(context)

        trainable_q_cps = theta[..., :-1].reshape(-1, self._n_trainable_q_pts, self.n_dim)
        trainable_t_scale = theta[..., -1:].reshape(-1)
        trainable_q_cps = trainable_q_cps * self.q_scale
        trainable_t_scale = trainable_t_scale * self.t_scale
        trainable_t_scale = torch.exp(trainable_t_scale)
        #middle_trainable_q_pts = torch.tanh(1000. * trainable_q_cps[:, :-1]) * np.pi
        middle_trainable_q_pts = 1000. * torch.tanh(trainable_q_cps[:, :-1]) * 2 * np.pi
        trainable_q_d = torch.tanh(trainable_q_cps[:, -1:]) * 2 * np.pi

        # w/ prior knowledge
        q_d = trainable_q_d + self.q_d_bias[None, None] - q_0
        # w/o prior knowledge
        #q_d = trainable_q_d

        q_cps = torch.cat([middle_trainable_q_pts, q_d], axis=-2)

        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_0, q_cps, trainable_t_scale, differentiable=True)

        self._traj_no += 1
        return q, q_dot, q_ddot, t, dt, duration