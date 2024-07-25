from copy import deepcopy
import torch
import mujoco
import numpy as np
from scipy import optimize as spo

from spline_rl.policy.bsmp_policy import BSMPPolicy
from spline_rl.utils.utils import unpack_data_box_pushing


class BSMPPolicyBoxPushing(BSMPPolicy):
    def __init__(self, env_info, dt, n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end,
                 t_scale=1., q_scale=1., q_d_scale=1., q_dot_d_scale=1., q_ddot_d_scale=1.):
        env_info['robot']['ee_desired_height'] = None
        super(BSMPPolicyBoxPushing, self).__init__(env_info, dt, n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end,
                                                   t_scale, q_scale, q_d_scale, q_dot_d_scale, q_ddot_d_scale)
        self.joint_pos_limit = env_info['robot']['joint_pos_limit']
        self._model = deepcopy(env_info['robot']['robot_model'])
        self._data = deepcopy(env_info['robot']['robot_data'])

        self._add_save_attr(
        )

    def unpack_context(self, context):
        q_0, q_dot_0, box_pos, box_rot, box_pos_d, box_rot_d = unpack_data_box_pushing(torch.tensor(context))
        q_ddot_0 = torch.zeros_like(q_0)
        return (q_0[:, None], q_dot_0[:, None], q_ddot_0[:, None],
                box_pos[:, None], box_rot[:, None], box_pos_d[:, None], box_rot_d[:, None])

    def ik_obj(self, q, desired_pos):
        for i in range(7):
            self._data.joint("panda_joint" + str(i + 1)).qpos = q[i]
        mujoco.mj_fwdPosition(self._model, self._data)
        pos = self._data.body("tcp").xpos
        diff_x = desired_pos - pos
        return np.linalg.norm(diff_x)

    def inverse_kinematics(self, pos, q0):
        self.bounds = spo.Bounds(*self.joint_pos_limit)
        options = {'maxiter': 300, 'ftol': 1e-06, 'iprint': 1, 'disp': False,
                'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None}
        r = spo.minimize(lambda x: self.ik_obj(x, pos), q0, method='SLSQP',
                        bounds=self.bounds, options=options)
        return r.x

    def compute_trajectory_from_theta(self, theta, context):
        q_0, q_dot_0, q_ddot_0, box_pos, box_rot, box_pos_d, box_rot_d = self.unpack_context(context)
        trainable_q_cps, trainable_t_cps = self.extract_qt(theta)
        trainable_t_cps = trainable_t_cps * self.t_scale
        middle_trainable_q_pts = torch.tanh(trainable_q_cps[:, :-1] * self.q_scale) * np.pi
        trainable_q_d = torch.tanh(trainable_q_cps[:, -1:] * self.q_d_scale) * np.pi

        q_d_s = []
        for k in range(q_0.shape[0]):
            q_d = self.inverse_kinematics(box_pos_d[k].detach().numpy(), q_0[k].detach().numpy())
            q_d_s.append(q_d)
        q_d_bias = torch.tensor(np.array(q_d_s))[:, None]
        q_d = q_d_bias + trainable_q_d 

        q1, q2, qm2, qm1 = self.compute_boundary_control_points_exp(trainable_t_cps, q_0, q_dot_0, q_ddot_0,
                                                                    q_d, torch.zeros_like(q_d), torch.zeros_like(q_d))
        q_begin = [q_0, q1, q2]
        q_end = [q_d, qm1, qm2]

        s = torch.linspace(0., 1., middle_trainable_q_pts.shape[1]+6)[None, 3:-3, None]
        q_b = q_0 * (1 - s) + q_d * s
        q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_b + middle_trainable_q_pts] + q_end[::-1], axis=-2)

        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps, trainable_t_cps, differentiable=True)

        self._traj_no += 1
        return q, q_dot, q_ddot, t, dt, duration

    def draw_action(self, state, policy_state=None):
        action, policy_state = super().draw_action(state, policy_state)
        return action, policy_state