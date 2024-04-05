import torch
import numpy as np
from scipy.interpolate import interp1d

from mp_pytorch.mp import ExpDecayPhaseGenerator, ProDMPBasisGenerator
from mushroom_rl.policy import Policy
from baseline.baseline_agent.optimizer import TrajectoryOptimizer
from utils.utils import unpack_data_airhockey



class ProDMPPolicy(Policy):
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

        phase_gn = ExpDecayPhaseGenerator()
        basis_gn = ProDMPBasisGenerator(
            phase_generator=phase_gn,
            num_basis=11 - 1,
            dt=1./1024)
        t = torch.linspace(0, self.dt * self.horizon, self.horizon).to(torch.float32)
        b = basis_gn.basis(t)
        db = basis_gn.vel_basis(t)

        #plt.subplot(121)
        #for i in range(11):
        #    plt.plot(t, b[:, i])
        #plt.plot(t, b.sum(-1), "--")
        #plt.subplot(122)
        #for i in range(11):
        #    plt.plot(t, db[:, i])
        #plt.show()

        self.N = b.detach().numpy()[None].astype(np.float64)
        self.dN = db.detach().numpy()[None].astype(np.float64)

        self.load_policy(env_info)

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

        trainable_q_cps = theta.reshape(-1, self._n_trainable_q_pts, self.n_dim)
        trainable_q_cps = trainable_q_cps / 50.
        middle_trainable_q_pts = torch.tanh(trainable_q_cps[:, :-1]) * np.pi
        trainable_q_d = torch.tanh(trainable_q_cps[:, -1:]) * np.pi

        x_des = np.array([1.31, 0., self.desired_ee_z])
        _, q_d_bias = self.optimizer.inverse_kinematics(x_des, q_0.detach().numpy()[0, 0])

        q_d = trainable_q_d + torch.tensor(q_d_bias)[None, None]

        q_cps = torch.cat([q_0, middle_trainable_q_pts, q_d], axis=-2)

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
