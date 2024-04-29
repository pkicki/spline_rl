import numpy as np
import torch
from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian
from baseline.baseline_agent.optimizer import TrajectoryOptimizer
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from mushroom_rl.policy import Policy

from spline_rl.policy.bsmp_policy import BSMPPolicy
from spline_rl.utils.bspline import BSpline
from spline_rl.utils.utils import unpack_data_airhockey


class BSMPPolicyStop(BSMPPolicy):
    def __init__(self, env_info, dt, n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end,
                 t_scale=1., q_scale=1., q_d_scale=1., q_dot_d_scale=1., q_ddot_d_scale=1.):
        super().__init__(env_info, dt, n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end,
                         t_scale, q_scale, q_d_scale, q_dot_d_scale, q_ddot_d_scale)
        self._n_trainable_q_stop_pts = self._n_q_pts - 5
        self._n_trainable_t_stop_pts = self._n_t_pts

        self._weights = np.zeros((self._n_trainable_q_pts * self.n_dim + self._n_trainable_q_stop_pts +
                                  self._n_trainable_t_pts + self._n_trainable_t_stop_pts,))


        self._add_save_attr(
            _n_trainable_q_stop_pts='primitive',
            _n_trainable_t_stop_pts='primitive',
        )

    def compute_trajectory_from_theta(self, theta, context):
        q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, puck, puck_dot = self.unpack_context(context)
        trainable_q_cps, trainable_q_stop_cps, trainable_t_cps, trainable_t_stop_cps = self.extract_qt(theta)
        trainable_t_cps = trainable_t_cps * self.t_scale
        middle_trainable_q_pts = torch.tanh(trainable_q_cps[:, :-3] * self.q_scale) * np.pi
        middle_trainable_q_stop_pts = torch.tanh(trainable_q_stop_cps[:, :-1] * self.q_scale) * np.pi
        trainable_q_d = torch.tanh(trainable_q_cps[:, -1:] * self.q_d_scale) * np.pi
        trainable_q_stop_d = torch.tanh(trainable_q_stop_cps[:, -1:] * self.q_d_scale) * np.pi
        trainable_q_dot_d = torch.tanh(trainable_q_cps[:, -2:-1] * self.q_dot_d_scale) * 2. * torch.tensor(self.joint_vel_limit)
        trainable_q_ddot_d = torch.tanh(trainable_q_cps[:, -3:-2] * self.q_ddot_d_scale) * torch.tensor(self.joint_acc_limit)


        dtau_dt = np.exp(self._t_bsp.N @ trainable_t_cps.detach().numpy())
        dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[-2]
        duration = np.sum(dt, axis=-1, keepdims=True)

        puck_pos = puck.detach().numpy()[:, :2]
        puck_dot = puck_dot.detach().numpy()[:, :2]
        expected_puck_pos = puck_pos + puck_dot * duration
        expected_puck_pos = np.clip(expected_puck_pos, [0.81, -0.35], [1.31, 0.35])
        puck_pos = expected_puck_pos
        goal = np.array([2.484, 0.])
        # Compute the vector that shoot the puck directly to the goal
        vec_puck_goal = (goal - puck_pos) / np.linalg.norm(goal - puck_pos)
        vec_puck_goal = np.concatenate([vec_puck_goal, np.zeros_like(vec_puck_goal)[..., -1:]], axis=-1)
        v_des = torch.tensor(vec_puck_goal)

        x_des = puck_pos# - (self.env_info['mallet']['radius'] + self.env_info['puck']['radius'] - 0.01) * v_des
        x_des = np.concatenate([x_des, np.ones_like(x_des)[..., -1:] * self.desired_ee_z], axis=-1)
        x_des = x_des.astype(np.float64)

        q_d_s = []
        for k in range(q_0.shape[0]):
            success, q_d = self.optimizer.solve_hit_config(x_des[k], v_des.detach().numpy()[k], q_0.detach().numpy()[k, 0])
            q_d_s.append(q_d)
        q_d_bias = torch.tensor(np.array(q_d_s))[:, None]
        q_d = q_d_bias + trainable_q_d 
        q_stop_d = q_d_bias + trainable_q_stop_d

        q_dot_d_s = []
        for k in range(q_0.shape[0]):
            q_dot_d = self.optimizer.solve_hitting_veocity(q_d.detach().numpy()[k, 0], v_des.detach().numpy()[k])
            q_dot_d_s.append(q_dot_d)
        q_dot_d = torch.tensor(np.array(q_dot_d_s))[:, None] + trainable_q_dot_d

        #trainable_q_ddot_d = torch.tanh(trainable_q_cps[:, -3:-2] / (10. * torch.sqrt(torch.abs(q_dot_d)))) * torch.tensor(self.joint_acc_limit)
        q_ddot_d = trainable_q_ddot_d
        q1, q2, qm2, qm1 = self.compute_boundary_control_points_exp(trainable_t_cps, q_0, q_dot_0, q_ddot_0,
                                                                    q_d, q_dot_d, q_ddot_d)
        q_begin = [q_0, q1, q2]
        q_end = [q_d, qm1, qm2]

        s = torch.linspace(0., 1., middle_trainable_q_pts.shape[1]+6)[None, 3:-3, None]
        q_b = q_0 * (1 - s) + q_d * s
        q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_b + middle_trainable_q_pts] + q_end[::-1], axis=-2)

        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps.to(torch.float32), trainable_t_cps.to(torch.float32), differentiable=True)

        # stopping trajectory
        q1, q2, qm2, qm1 = self.compute_boundary_control_points_exp(trainable_t_stop_cps, q_d, q_dot_d, q_ddot_d,
                                                                    q_stop_d, torch.zeros_like(q_dot_d), torch.zeros_like(q_ddot_d))
        q_begin = [q_d, q1, q2]
        q_end = [q_stop_d, qm1, qm2]
        q_stop_b = q_d * (1 - s) + q_stop_d * s
        q_stop_cps = torch.cat(q_begin + [q_stop_b + middle_trainable_q_stop_pts] + q_end[::-1], axis=-2)

        q_stop, q_stop_dot, q_stop_ddot, t_stop, dt_stop, duration_stop = self.compute_trajectory(q_stop_cps.to(torch.float32),
                                                                                                  trainable_t_stop_cps.to(torch.float32),
                                                                                                  differentiable=True)
        q = torch.cat([q, q_stop[:, 1:]], axis=-2)
        q_dot = torch.cat([q_dot, q_stop_dot[:, 1:]], axis=-2)
        q_ddot = torch.cat([q_ddot, q_stop_ddot[:, 1:]], axis=-2)
        t = torch.cat([t, t[:, -1:] + t_stop[:, 1:]], axis=-1)
        dt = torch.cat([dt, dt_stop[:, 1:]], axis=-1)
        duration = duration + duration_stop

        self._traj_no += 1
        return q, q_dot, q_ddot, t, dt, duration

    def extract_qt(self, x):
        # TODO: make it suitable for parallel envs
        if len(x.shape) == 1:
            x = x[None]
        idx_q = self._n_trainable_q_pts * self.n_dim
        idx_q_stop = idx_q + self._n_trainable_q_stop_pts * self.n_dim
        idx_t = idx_q_stop + self._n_trainable_t_pts
        q_cps = x[:, :idx_q]
        q_stop_cps = x[:, idx_q:idx_q_stop]
        t_cps = x[:, idx_q_stop:idx_t]
        t_stop_cps = x[:, idx_t:]
        q_cps = q_cps.reshape(-1, self._n_trainable_q_pts, self.n_dim)
        q_stop_cps = q_stop_cps.reshape(-1, self._n_trainable_q_stop_pts, self.n_dim)
        t_cps = t_cps.reshape(-1, self._n_trainable_t_pts, 1)
        t_stop_cps = t_stop_cps.reshape(-1, self._n_trainable_t_stop_pts, 1)
        return q_cps, q_stop_cps, t_cps, t_stop_cps
