import numpy as np
from spline_rl.policy.bsmp_policy_kino import BSMPPolicyKino
import torch
from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian
from baseline.baseline_agent.optimizer import TrajectoryOptimizer
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from mushroom_rl.policy import Policy

from spline_rl.utils.bspline import BSpline
from spline_rl.utils.utils import unpack_data_kinodynamic


class BSMPPolicyUnstructuredKino(BSMPPolicyKino):
    def compute_trajectory_from_theta(self, theta, context):
        q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d = self.unpack_context(context)
        trainable_q_cps, trainable_t_cps = self.extract_qt(theta)
        trainable_t_cps = trainable_t_cps * self.t_scale
        trainable_q_middle = trainable_q_cps[:, :-1] * self.q_scale
        trainable_q_d_ = trainable_q_cps[:, -1:] * self.q_d_scale
        trainable_q_pts = torch.tanh(trainable_q_middle) * 2. * np.pi
        trainable_q_d = torch.tanh(trainable_q_d_) * 2. * np.pi

        #q_d_bias = torch.tensor([-0.1178,  0.2472, -2.1531,  2.0209, -2.6691, -0.7095,  0.9851])[None, None]
        q_d = trainable_q_d + q_0#q_d_bias

        q1, q2, qm2, qm1 = self.compute_boundary_control_points_exp(trainable_t_cps, q_0, q_dot_0, q_ddot_0,
                                                                    q_d, q_dot_d, q_ddot_d)
        q_begin = [q_0, q1, q2]
        q_end = [q_d, qm1, qm2]

        s = torch.linspace(0., 1., trainable_q_pts.shape[1]+6)[None, 3:-3, None]
        q_b = q_0 * (1 - s) + q_d * s
        q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_b + trainable_q_pts] + q_end[::-1], axis=-2)

        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps.to(torch.float32), trainable_t_cps.to(torch.float32), differentiable=True)

        self._traj_no += 1
        return q, q_dot, q_ddot, t, dt, duration