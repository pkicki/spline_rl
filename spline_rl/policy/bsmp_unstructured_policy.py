import torch
import numpy as np

from spline_rl.policy.bsmp_policy import BSMPPolicy


class BSMPUnstructuredPolicy(BSMPPolicy):
    def compute_trajectory_from_theta(self, theta, context):
        q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, puck, puck_dot = self.unpack_context(context)
        trainable_q_cps, trainable_t_cps = self.extract_qt(theta)
        trainable_q_cps = trainable_q_cps * self.q_scale
        trainable_t_cps = trainable_t_cps * self.t_scale
        middle_trainable_q_pts = torch.tanh(trainable_q_cps[:, :-1]) * np.pi
        trainable_q_d = torch.tanh(trainable_q_cps[:, -1:]) * np.pi

        x_des = np.array([1.31, 0., self.desired_ee_z])
        _, q_d_bias = self.optimizer.inverse_kinematics(x_des, q_0.detach().numpy()[0, 0])

        q_d = trainable_q_d + torch.tensor(q_d_bias)[None, None]
        q_dot_d = torch.zeros_like(q_d)
        q_ddot_d = torch.zeros_like(q_d)
        q1, q2, qm2, qm1 = self.compute_boundary_control_points_exp(trainable_t_cps, q_0, q_dot_0, q_ddot_0,
                                                                    q_d, q_dot_d, q_ddot_d)
        q_begin = [q_0, q1, q2]
        q_end = [q_d, qm1, qm2]

        s = torch.linspace(0., 1., middle_trainable_q_pts.shape[1]+6)[None, 3:-3, None]
        q_b = q_0 * (1 - s) + q_d * s
        q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_b + middle_trainable_q_pts] + q_end[::-1], axis=-2)

        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps, trainable_t_cps, differentiable=True)
        self._traj_no += 1
        return q, q_dot, q_ddot, t, dt, duration
