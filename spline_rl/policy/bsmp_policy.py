import torch
import numpy as np
from scipy.interpolate import interp1d

from mushroom_rl.policy import Policy

from spline_rl.utils.bspline import BSpline


class BSMPPolicy(Policy):
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

        self.episode_duration = env_info['episode_duration']
        self.joint_vel_limit = env_info['robot']['joint_vel_limit'][1]
        self.joint_acc_limit = env_info['robot']['joint_acc_limit'][1]

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
            episode_duration='primitive',
            joint_vel_limit='pickle',
            joint_acc_limit='pickle',
            t_scale='primitive',
            q_scale='primitive',
            q_d_scale='primitive',
            q_dot_d_scale='primitive',
            q_ddot_d_scale='primitive',
        )

    def unpack_context(self, context):
        raise NotImplementedError

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
            #for i in range(q.shape[-1]):
            #    plt.plot(t[0], q[0, :, i], label=f'q_{i}')
            #plt.legend()
            #plt.show()
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

        def compute_duration(dtau_dt):
            dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[-2]
            t = np.cumsum(dt, axis=-1) - dt[..., :1] if not differentiable else torch.cumsum(dt, dim=-1) - dt[..., :1]
            duration = t[:, -1]
            return t, dt, duration

        dtau_dt = torch.exp(tN @ t_cps) if differentiable else np.exp(tN @ t_cps)
        #ddtau_dtt = dtau_dt * (tdN @ t_cps)
        ddtau_dtt = dtau_dt**2 * (tdN @ t_cps)

        t, dt, duration = compute_duration(dtau_dt)

        duration = torch.tile(duration[:, None, None], (1, dtau_dt.shape[1], 1))
        too_long_trajectory = duration > self.episode_duration
        c = duration / self.episode_duration
        log_c = torch.log(c)
        tN_inv = torch.linalg.pinv(tN)
        t_cps_update = tN_inv @ log_c
        t_cps_ = t_cps + t_cps_update
        dtau_dt_ = torch.exp(tN @ t_cps_) if differentiable else np.exp(tN @ t_cps_)
        ddtau_dtt_ = dtau_dt_**2 * (tdN @ t_cps_)

        dtau_dt = torch.where(too_long_trajectory, dtau_dt_, dtau_dt)
        ddtau_dtt = torch.where(too_long_trajectory, ddtau_dtt_, ddtau_dtt)
        #t_, dt_, duration_ = compute_duration(dtau_dt_)
        t, dt, duration = compute_duration(dtau_dt)

        #ddtau_dtt__ = (dtau_dt_[:, 1:] - dtau_dt_[:, :-1]) / (t_[:, 1:] - t_[:, :-1])[..., None]
        #ddtau_dtt_hand = (dtau_dt[:, 1:] - dtau_dt[:, :-1]) / (t[:, 1:] - t[:, :-1])[..., None]
        ##ddtau_dtt__ = (dtau_dt_[:, 1:] - dtau_dt_[:, :-1]) / (1. / dtau_dt.shape[-2])
        ##ddtau_dtt_hand = (dtau_dt[:, 1:] - dtau_dt[:, :-1]) / (1. / dtau_dt.shape[-2])
        
        #plt.subplot(311)
        #plt.plot(t[0], dtau_dt[0, :, 0], label='dtau_dt')
        #plt.plot(t_[0], dtau_dt_[0, :, 0], label='dtau_dt_scaled')
        #plt.subplot(312)
        #plt.plot(t[0], ddtau_dtt[0, :, 0], label='ddtau_dtt')
        #plt.plot(t[0, 1:], ddtau_dtt_hand[0, :, 0], label='ddtau_dtt_hand')
        #plt.plot(t_[0], ddtau_dtt_[0, :, 0], label='ddtau_dtt_')
        #plt.legend()
        #plt.subplot(313)
        #plt.plot(t_[0], ddtau_dtt[0, :, 0], label='ddtau_dtt')
        #plt.plot(t_[0], ddtau_dtt_[0, :, 0], label='ddtau_dtt_')
        #plt.plot(t_[0, 1:], ddtau_dtt__[0, :, 0], label='ddtau_dtt_hand')
        #plt.legend()
        #plt.show()

        q_dot = q_dot_tau * dtau_dt
        q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt
        return q, q_dot, q_ddot, t, dt, duration
