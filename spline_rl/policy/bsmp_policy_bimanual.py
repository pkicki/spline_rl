import numpy as np
import torch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.utils import inverse_kinematics as ik

from spline_rl.policy.bsmp_policy import BSMPPolicy
from spline_rl.utils.bspline import BSpline
from spline_rl.utils.utils import unpack_data_bimanual


class BSMPPolicyBimanual(BSMPPolicy):
    def __init__(self, env_info, dt, n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end,
                 t_scale=1., q_scale=1., q_d_scale=1., q_dot_d_scale=1., q_ddot_d_scale=1.):
        super().__init__(env_info, dt, n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end,
                         t_scale, q_scale, q_d_scale, q_dot_d_scale, q_ddot_d_scale)
        
        self._model = env_info['model']
        self._data = env_info['data']
        self._dm_physics = env_info['dm_physics']
        self._robot_joint_ids = env_info['robot']['joint_ids']
        self._left_robot_joint_ids = env_info['robot']['left_joint_ids']
        self._right_robot_joint_ids = env_info['robot']['right_joint_ids']

        #self.q_d_bias = torch.tensor([
        #    0., # base joint
        #    2.2220e-01, 4.9374e-01,  5.6134e-01, -1.7766e-01, -9.0099e-03,  3.1140e-01, # left arm
        #    -5.3729e-02, -6.5654e-01, -5.3388e-01,  2.0437e-01, -1.3195e-02, -7.5073e-02]) # right arm

        self.q_d_bias = torch.tensor([
            0.,
            0.21646051,  0.48360129,  0.55410618, -0.17025892, -0.00847046,  0.30138272,
            -0.04727616, -0.64639233, -0.52647715, 0.1963449 , -0.01348486, -0.06694195])

        self.q_dm3_bias = torch.tensor([
            0.,
            0.20561048, 0.46379331, 0.53974225, -0.15465164, -0.006246, 0.28681292,
            -0.03560051, -0.6268268 , -0.51194737, 0.17966371, -0.01396815, -0.05032922])

        self.q_dot_d = np.array([
            0.,
            1.0945e+00,  1.8589e+00,  1.7170e+00, -2.0528e-01, -2.6921e-01,  0.,
            -1.2133e+00, -1.7710e+00, -1.7345e+00,  1.8163e-01,  2.5958e-01, 0.])

        self._add_save_attr(
            q_d_bias='pickle',
        )

    def compute_bias(self, q_0):
        q_0[..., 0] = 0.
        #pos_left = np.array([-0.325, 0.6, 0.075])
        #pos_right = np.array([0.175, 0.6, 0.075])
        pos_left = np.array([-0.325, 0.6, 0.1])
        pos_right = np.array([0.175, 0.6, 0.1])
        rot_left = np.array([0., 0., 1., 0.])
        rot_right = np.array([0., 0., 1., 0.])
        self._data.qpos[self._robot_joint_ids] = q_0
        self._dm_physics.data.qpos[:] = self._data.qpos[:]
        def arm_ik(robot, pos, rot):
            return ik.qpos_from_site_pose(self._dm_physics, f"EE_ur5{robot}", pos, rot,
                            joint_names=[f"joint{i}_ur5{robot}" for i in range(6)])
        result_left = arm_ik("left", pos_left, rot_left)
        result_right = arm_ik("right", pos_right, rot_right)
        if result_left.success and result_right.success:
            q_left = result_left.qpos[self._robot_joint_ids[1:7]]
            q_right = result_right.qpos[self._robot_joint_ids[7:]]
            return np.concatenate([np.zeros_like(q_left[..., :1]), q_left, q_right], axis=-1)
        return None

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

        #q_dot_d = 0.2 * torch.tensor(self.q_dot_d)[None, None]
        #q_ddot_d = -30. * q_dot_d

        #q_d_bias = self.q_d_bias
        #q_bias_ = self.compute_bias(q_0)
        #q_d = trainable_q_d + q_0#q_d_bias
        q_d = trainable_q_d + self.q_d_bias[None, None]

        # Jacobian computations
        def get_q_dot_d(robot):
            assert robot in ["left", "right"]
            site_id = self._data.site(f"EE_ur5{robot}").id
            jacp = np.zeros((3, self._model.nv))
            jacr = np.zeros((3, self._model.nv))
            mujoco.mj_jacSite(self._model, self._data, jacp, jacr, site_id)
            J = jacp[:, self._left_robot_joint_ids] if robot == "left" else jacp[:, self._right_robot_joint_ids]
            pinvJ = np.linalg.pinv(J)
            q_dot_d = pinvJ @ np.array([0., 0., -1.])
            return q_dot_d
        self._data.qpos[self._robot_joint_ids] = q_d
        mujoco.mj_fwdPosition(self._model, self._data)
        left_q_dot_d = get_q_dot_d("left")
        right_q_dot_d = get_q_dot_d("right")

        q_dot_d = 0.2 * torch.tensor(np.concatenate([[0.], left_q_dot_d, right_q_dot_d]))[None, None]
        q_ddot_d = -30. * q_dot_d

        q1, q2, qm2, qm1 = self.compute_boundary_control_points_exp(trainable_t_cps, q_0, q_dot_0, q_ddot_0,
                                                                    q_d, q_dot_d, q_ddot_d)
        q_begin = [q_0, q1, q2]
        q_end = [q_d, qm1, qm2]

        #s = torch.linspace(0., 1., trainable_q_pts.shape[1]+6)[None, 3:-4, None]
        #q_b = q_0 * (1 - s) + q_d * s
        #q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_b + trainable_q_pts[..., :-1, :]] + [self.q_dm3_bias[None, None] + trainable_q_pts[..., -1:, :]] + q_end[::-1], axis=-2)

        #s = torch.linspace(0., 1., trainable_q_pts.shape[1]+6)[None, 3:-3, None]
        #q_b = q_0 * (1 - s) + q_d * s
        s = torch.linspace(0., 1., trainable_q_pts.shape[1]+2)[None, 1:-1, None]
        q_b = q_begin[-1] * (1 - s) + q_end[-1] * s
        q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_b + trainable_q_pts] + q_end[::-1], axis=-2)

        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps, trainable_t_cps, differentiable=True)

        self._traj_no += 1
        return q, q_dot, q_ddot, t, dt, duration