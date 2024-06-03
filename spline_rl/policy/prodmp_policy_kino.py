import numpy as np
from spline_rl.policy.prodmp_policy import ProDMPPolicy
import torch

from mushroom_rl.features._implementations.basis_features import BasisFeatures
from spline_rl.utils.utils import unpack_data_kinodynamic

import matplotlib.pyplot as plt


class ProDMPPolicyKino(ProDMPPolicy):
    def unpack_context(self, context):
        if context is None:
            raise NotImplementedError
        else:
            q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d = unpack_data_kinodynamic(torch.tensor(context))
        return q_0[:, None], q_d[:, None], q_dot_0[:, None], q_dot_d[:, None], q_ddot_0[:, None], q_ddot_d[:, None]

    def compute_trajectory_from_theta(self, theta, context):
        q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d = self.unpack_context(context)

        trainable_q_cps = theta[..., :-1].reshape(-1, self._n_trainable_q_pts, self.n_dim)
        trainable_t_scale = theta[..., -1:].reshape(-1)
        trainable_q_cps = trainable_q_cps * self.q_scale
        trainable_t_scale = trainable_t_scale * self.t_scale
        trainable_t_scale = torch.exp(trainable_t_scale)
        #middle_trainable_q_pts = torch.tanh(1000. * trainable_q_cps[:, :-1]) * np.pi
        middle_trainable_q_pts = 1000. * torch.tanh(trainable_q_cps[:, :-1]) * 2 * np.pi
        trainable_q_d = torch.tanh(trainable_q_cps[:, -1:]) * 2 * np.pi

        #q_d = trainable_q_d + q_d - q_0
        q_d = trainable_q_d

        q_cps = torch.cat([middle_trainable_q_pts, q_d], axis=-2)

        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_0, q_cps, trainable_t_scale, differentiable=True)

        self._traj_no += 1
        return q, q_dot, q_ddot, t, dt, duration