import numpy as np
import torch
from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian
from baseline.baseline_agent.optimizer import TrajectoryOptimizer
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from mushroom_rl.policy import Policy

from spline_rl.utils.bspline import BSpline
from spline_rl.utils.utils import unpack_data_kinodynamic


class BSMPPolicyKino(Policy):
    def __init__(self, env_info, dt, n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end,
                 t_scale=1., q_scale=1., q_d_scale=1., q_dot_d_scale=1., q_ddot_d_scale=1.):
        self.dt = dt
        self.n_dim = n_dim
        self._n_q_pts = n_q_pts
        self._n_t_pts = n_t_pts
        self._n_pts_fixed_begin = n_pts_fixed_begin
        self._n_pts_fixed_end = n_pts_fixed_end
        self._n_trainable_q_pts = self._n_q_pts - (self._n_pts_fixed_begin + self._n_pts_fixed_end)
        self._n_trainable_t_pts = self._n_t_pts

        self._q_bsp = BSpline(self._n_q_pts)
        self._t_bsp = BSpline(self._n_t_pts)
        self._qdd1 = self._q_bsp.ddN[0, 0, 0]
        self._qdd2 = self._q_bsp.ddN[0, 0, 1]
        self._qdd3 = self._q_bsp.ddN[0, 0, 2]
        self._qd1 = self._q_bsp.dN[0, 0, 1]
        self._td1 = self._t_bsp.dN[0, 0, 1]

        self.t_scale = t_scale
        self.q_scale = q_scale
        self.q_d_scale = q_d_scale
        self.q_dot_d_scale = q_dot_d_scale
        self.q_ddot_d_scale = q_ddot_d_scale

        self.q = None
        self.q_dot = None
        self.q_ddot = None
        self.duration = None

        self._weights = np.zeros((self._n_trainable_q_pts * self.n_dim + self._n_trainable_t_pts,))

        self._traj_no = 0

        self.optimizer = None
        self.load_policy(env_info)

        policy_state_shape = (1,)
        super().__init__(policy_state_shape)

        self._add_save_attr(
            dt='primitive',
            n_dim='primitive',
            _n_q_pts='primitive',
            _n_t_pts='primitive',
            _n_pts_fixed_begin='primitive',
            _n_pts_fixed_end='primitive',
            _n_trainable_q_pts='primitive',
            _n_trainable_t_pts='primitive',
            _q_bsp='pickle',
            _t_bsp='pickle',
            _qdd1='primitive',
            _qdd2='primitive',
            _qdd3='primitive',
            _qd1='primitive',
            _td1='primitive',
            _traj_no='primitive',
        )

    def load_policy(self, env_info):
        self.optimizer = TrajectoryOptimizer(env_info)

    def unpack_context(self, context):
        if context is None:
            raise NotImplementedError
        else:
            q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d = unpack_data_kinodynamic(torch.tensor(context))
        return q_0[:, None], q_d[:, None], q_dot_0[:, None], q_dot_d[:, None], q_ddot_0[:, None], q_ddot_d[:, None]

    def compute_trajectory_from_theta(self, theta, context):
        q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d = self.unpack_context(context)
        trainable_q_cps, trainable_t_cps = self.extract_qt(theta)
        trainable_t_cps = trainable_t_cps * self.t_scale
        trainable_q_pts = torch.tanh(trainable_q_cps * self.q_scale) * 2. * np.pi

        #q_d = q_0

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
            duration = duration.detach().numpy()
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
        action = np.stack([q, q_dot, q_ddot], axis=-2) 
        action = torch.tensor(action)
        return action, policy_state

    def extract_qt(self, x):
        # TODO: make it suitable for parallel envs
        if len(x.shape) == 1:
            x = x[None]
        q_cps = x[:, :self._n_trainable_q_pts * self.n_dim]
        t_cps = x[:, self._n_trainable_q_pts * self.n_dim:]
        q_cps = q_cps.reshape(-1, self._n_trainable_q_pts, self.n_dim)
        t_cps = t_cps.reshape(-1, self._n_trainable_t_pts, 1)
        return q_cps, t_cps

    def set_weights(self, weights):
        self._weights = weights

    def get_weights(self):
        return self._weights

    def compute_boundary_control_points(self, dtau_dt, q0, q_dot_0, q_ddot_0, qd, q_dot_d, q_ddot_d):
        q1 = q_dot_0 / dtau_dt[:, :1] / self._qd1 + q0
        qm1 = qd - q_dot_d / dtau_dt[:, -1:] / self._qd1
        q2 = ((q_ddot_0 / dtau_dt[:, :1] -
               self._qd1 * self._td1 * (q1 - q0) * (dtau_dt[:, 1] - dtau_dt[:, 0])[:, None]) / dtau_dt[:, :1]
              - self._qdd1 * q0 - self._qdd2 * q1) / self._qdd3
        qm2 = ((q_ddot_d / dtau_dt[:, -1:] -
                self._qd1 * self._td1 * (qd - qm1) * (dtau_dt[:, -1] - dtau_dt[:, -2])[:, None]) / dtau_dt[:, -1:]
               - self._qdd1 * qd - self._qdd2 * qm1) / self._qdd3
        return q1, q2, qm2, qm1

    def compute_boundary_control_points_exp(self, dtau_dt, q0, q_dot_0, q_ddot_0, qd, q_dot_d, q_ddot_d):
        q1 = q_dot_0 / (torch.exp(dtau_dt[:, :1]) * self._qd1) + q0
        qm1 = qd - q_dot_d / (torch.exp(dtau_dt[:, -1:]) * self._qd1)
        q2 = (q_ddot_0 / torch.exp(dtau_dt[:, :1])**2
              - self._qd1 * self._td1 * (q1 - q0) * (dtau_dt[:, 1] - dtau_dt[:, 0])[:, None]
              - self._qdd1 * q0
              - self._qdd2 * q1) / self._qdd3
        qm2 = (q_ddot_d / torch.exp(dtau_dt[:, -1:])**2
               - self._qd1 * self._td1 * (qd - qm1) * (dtau_dt[:, -1] - dtau_dt[:, -2])[:, None]
               - self._qdd1 * qd
               - self._qdd2 * qm1) / self._qdd3
        return q1, q2, qm2, qm1


    def compute_trajectory(self, q_cps, t_cps, differentiable=False):
        qN = self._q_bsp.N
        qdN = self._q_bsp.dN
        qddN = self._q_bsp.ddN
        tN = self._t_bsp.N
        tdN = self._t_bsp.dN
        if differentiable:
            qN = torch.tensor(qN)
            qdN = torch.tensor(qdN)
            qddN = torch.tensor(qddN) 
            tN = torch.tensor(tN)
            tdN = torch.tensor(tdN)

        q = qN @ q_cps
        q_dot_tau = qdN @ q_cps
        q_ddot_tau = qddN @ q_cps

        dtau_dt = torch.exp(tN @ t_cps) if differentiable else np.exp(tN @ t_cps)
        ddtau_dtt = dtau_dt * (tdN @ t_cps)

        dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[-2]
        t = np.cumsum(dt, axis=-1) - dt[..., :1] if not differentiable else torch.cumsum(dt, dim=-1) - dt[..., :1]
        duration = t[:, -1]

        q_dot = q_dot_tau * dtau_dt
        q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt
        return q, q_dot, q_ddot, t, dt, duration
