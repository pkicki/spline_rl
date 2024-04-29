import numpy as np
import torch
from scipy.interpolate import interp1d

from mushroom_rl.policy import Policy
from baseline.baseline_agent.optimizer import TrajectoryOptimizer

from mushroom_rl.features._implementations.basis_features import BasisFeatures
from mushroom_rl.features.basis import GaussianRBF
from spline_rl.utils.gaussian_derivative import dGaussianRBF
from spline_rl.utils.utils import unpack_data_airhockey


class ProMPPolicy(Policy):
    def __init__(self, env_info, n_q_pts, n_dim, n_pts_fixed_begin=1):
        self.n_dim = n_dim
        self._n_q_pts = n_q_pts
        self._n_pts_fixed_begin = n_pts_fixed_begin
        self._n_trainable_q_pts = self._n_q_pts - self._n_pts_fixed_begin

        self._traj_no = 0

        self.desired_ee_z = env_info['robot']['ee_desired_height']
        self.joint_vel_limit = env_info['robot']['joint_vel_limit'][1]
        self.joint_acc_limit = env_info['robot']['joint_acc_limit'][1]

        self.dt = env_info['dt']
        self.horizon = env_info['rl_info'].horizon

        self.phi = BasisFeatures(GaussianRBF.generate([self._n_q_pts], [0.], [1.], eta=0.95))
        self.dphi = BasisFeatures(dGaussianRBF.generate([self._n_q_pts], [0.], [1.], eta=0.95))
        self.N = np.stack([self.phi(i) for i in np.linspace(0, 1, self.horizon)], axis=0)[None]
        self.dN = np.stack([self.dphi(i) for i in np.linspace(0, 1, self.horizon)], axis=0)[None]

        sum = np.sum(self.N, axis=-1, keepdims=True)
        dN = self.dN
        dsum = np.sum(dN, axis=-1, keepdims=True)

        self.dN = (dN * sum - self.N * dsum) / sum**2
        self.N = self.N / sum
        dN_ = np.diff(self.N, axis=1)

        #plt.subplot(221)
        #for i in range(11):
        #    plt.plot(self.N[0, :, i])
        #plt.subplot(222)
        #plt.plot(np.sum(self.N[0], axis=-1))
        #plt.subplot(223)
        #for i in range(11):
        #    plt.plot(self.dN[0, :, i])
        #for i in range(11):
        #    plt.plot(dN_[0, :, i] * 150, '--')
        #plt.subplot(224)
        #plt.plot(np.sum(self.dN[0], axis=-1))
        #plt.show()

        self.load_policy(env_info)

        x_des = np.array([1.31, 0., self.desired_ee_z])
        #q_0 = np.array([-7.16000830e-06,  6.97494070e-01,  7.26955352e-06, -5.04898567e-01,
        #6.60813111e-07,  1.92857916e+00,  0.00000000e+00])
        #q_d = np.array([-5.76620323e-06,  9.46021120e-01,  7.25893860e-06, -2.91822736e-01,
        #7.06271697e-07,  1.36568904e+00, -6.34735204e-20])
        q_0 = np.array([-7.1600e-06,  6.9749e-01,  7.2696e-06, -5.0490e-01,  6.6081e-07,
           1.9286e+00,  0.0000e+00])
        q_d = np.array([-3.72483757e-06,  1.23533666e+00,  7.29216661e-06, -1.33108648e-01,
        7.61155450e-07,  8.73456051e-01,  7.72071616e-20])
        s = np.linspace(0., 1., self._n_q_pts)
        sh = np.linspace(0., 1., self.horizon)
        qs = q_0[None] * (1 - s[:, None]) + q_d[None] * s[:, None]
        qref = q_0[None] * (1 - sh[:, None]) + q_d[None] * sh[:, None] 
        qref = torch.tensor(qref)
        qs_trainable = torch.tensor(qs, requires_grad=True)
        opt = torch.optim.Adam([qs_trainable], lr=0.01)
        for i in range(1000):
            opt.zero_grad()
            q = torch.tensor(self.N[0]) @ qs_trainable
            loss = torch.sum((q - qref)**2)
            loss.backward()
            opt.step()
            #print(loss)

        self.q_bias = qs_trainable
        #for i in range(6):
        #    plt.subplot(231+i)
        #    plt.plot(q.detach().numpy()[:, i])
        #    plt.plot(qref.detach().numpy()[:, i])
        #plt.show()

        policy_state_shape = (1,)
        super().__init__(policy_state_shape)

        self._add_save_attr(
            dt='primitive',
            horizon='primitive',
            n_dim='primitive',
            q_bias='pickle',
            _n_q_pts='primitive',
            _n_pts_fixed_begin='primitive',
            _n_trainable_q_pts='primitive',
            _traj_no='primitive',
            desired_ee_z='primitive',
            joint_vel_limit='pickle',
            joint_acc_limit='pickle',
        )
    
    def load_policy(self, env_info):
        self.optimizer = TrajectoryOptimizer(env_info)

    def unpack_context(self, context):
        if context is None:
            raise NotImplementedError
        else:
            puck, puck_dot, q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, opponent_mallet = unpack_data_airhockey(torch.tensor(context))
        return q_0[:, None], q_d[:, None], q_dot_0[:, None], q_dot_d[:, None], q_ddot_0[:, None], q_ddot_d[:, None], puck

    def compute_trajectory_from_theta(self, theta, context):
        q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, puck = self.unpack_context(context)

        x_des = np.array([1.31, 0., self.desired_ee_z])
        _, q_d_bias = self.optimizer.inverse_kinematics(x_des, q_0.detach().numpy()[0, 0])

        trainable_q_cps = theta.reshape(-1, self._n_trainable_q_pts, self.n_dim)
        trainable_q_cps = trainable_q_cps / 50.
        trainable_q = torch.tanh(trainable_q_cps) * np.pi

        N0 = torch.tensor(self.N[:, 0])
        q_cps_n0 = trainable_q + self.q_bias[None, 1:]
        #q_cps_n0 = self.q_bias[None, 1:]

        q_cps_0 = (q_0[:, 0] - N0[0, 1:] @ q_cps_n0) / N0[0, 0]
        q_cps = torch.cat([q_cps_0[:, None], q_cps_n0], axis=-2)

        #q = self.N @ q_cps.detach().numpy()

        #for i in range(6):
        #    plt.subplot(231+i)
        #    plt.plot(q[0, :, i])
        #    plt.plot([0], q_0[0, :, i], 'gx')
        #    #plt.plot([150], q_d[0, :, i], 'rx')
        #plt.show()

        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps, differentiable=True)
        #q_dot_scale = (torch.abs(q_dot) / torch.tensor(self.joint_vel_limit))
        #q_ddot_scale = (torch.abs(q_ddot) / torch.tensor(self.joint_acc_limit))
        #q_dot_scale_max = torch.amax(q_dot_scale, (-2, -1), keepdim=True)
        #q_ddot_scale_max = torch.amax(q_ddot_scale, (-2, -1), keepdim=True)
        #scale_max = torch.maximum(q_dot_scale_max, q_ddot_scale_max**(1./2))
        #trainable_t_cps -= torch.log(scale_max)
        #q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps.to(torch.float32), trainable_t_cps.to(torch.float32), differentiable=True)

        #q_ = q.detach().numpy()[0]
        #q_dot_ = q_dot.detach().numpy()[0]
        #q_ddot_ = q_ddot.detach().numpy()[0]
        #t_ = t.detach().numpy()[0]
        #qdl = self.joint_vel_limit
        #qddl = self.joint_acc_limit
        #for i in range(self.n_dim):
        #    plt.subplot(3, 7, 1+i)
        #    plt.plot(t_, q_[:, i])
        #    plt.subplot(3, 7, 1+i+self.n_dim)
        #    plt.plot(t_, q_dot_[:, i])
        #    plt.plot([t_[0], t_[-1]], [qdl[i], qdl[i]], 'r--')
        #    plt.plot([t_[0], t_[-1]], [-qdl[i], -qdl[i]], 'r--')
        #    plt.subplot(3, 7, 1+i+2*self.n_dim)
        #    plt.plot(t_, q_ddot_[:, i])
        #    plt.plot([t_[0], t_[-1]], [qddl[i], qddl[i]], 'r--')
        #    plt.plot([t_[0], t_[-1]], [-qddl[i], -qddl[i]], 'r--')
        #plt.show()

        #xyz = []
        #for k in range(q.shape[1]):
        #    xyz_ = self.optimizer.forward_kinematics(q.detach().numpy()[0, k])
        #    xyz.append(xyz_)
        #xyz = np.array(xyz)
        #plt.subplot(121)
        #plt.plot(xyz[:, 0], xyz[:, 1])
        #plt.subplot(122)
        #plt.plot(xyz[:, 2])
        #plt.show()

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
            #q_ddot = q_ddot.detach().numpy()
            t = t.detach().numpy()
            #duration = duration.detach().numpy()
            self.q = interp1d(t[0], q[0], axis=0)
            self.q_dot = interp1d(t[0], q_dot[0], axis=0)
            #self.q_ddot = interp1d(t[0], q_ddot[0], axis=0)
            #self.duration = duration[0]
            self.duration = duration
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
            #q_ddot = self.q_ddot(t)
        else:
            q = self.q(self.duration)
            q_dot = np.zeros_like(q)
            #q_ddot = np.zeros_like(q)
        policy_state[0] += 1
        #action = np.stack([q, q_dot, q_ddot], axis=-2) 
        q_dot = np.clip(q_dot, -self.joint_vel_limit, self.joint_vel_limit)
        action = np.stack([q, q_dot], axis=-2) 
        action = torch.tensor(action, dtype=torch.float32)
        return action, torch.tensor(policy_state)

    def set_weights(self, weights):
        self._weights = weights

    def compute_trajectory(self, q_cps, differentiable=False):
        N = self.N
        dN = self.dN
        if differentiable:
            N = torch.tensor(N)
            dN = torch.tensor(dN)

        q = N @ q_cps
        q_dot = dN @ q_cps
        q_ddot = torch.zeros_like(q_dot)

        duration_ = self.dt * self.horizon
        duration = duration_ * torch.ones((q_cps.shape[0], 1))
        t = torch.linspace(0., duration_, N.shape[1])[None].repeat((q_cps.shape[0], 1))#[..., None]
        dt = self.dt * torch.ones_like(t)

        return q, q_dot, q_ddot, t, dt, duration
