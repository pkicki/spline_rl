import numpy as np
from spline_rl.policy.mp_policy import MPPolicy
import torch
from scipy.interpolate import interp1d

from mushroom_rl.policy import Policy
from baseline.baseline_agent.optimizer import TrajectoryOptimizer

from mushroom_rl.features._implementations.basis_features import BasisFeatures
from mushroom_rl.features.basis import GaussianRBF
from spline_rl.utils.gaussian_derivative import dGaussianRBF
from spline_rl.utils.utils import unpack_data_airhockey


class ProMPPolicy(MPPolicy):
    def __init__(self, env_info, n_q_cps, n_dim, n_pts_fixed_begin=1,
                 t_scale=1., q_scale=1., q_d_scale=1., q_dot_d_scale=1., q_ddot_d_scale=1., **kwargs):
        super().__init__(env_info, n_q_cps, n_dim, n_pts_fixed_begin,
                         t_scale, q_scale, q_d_scale, q_dot_d_scale, q_ddot_d_scale)
        q_0 = np.array([-7.1600e-06,  6.9749e-01,  7.2696e-06, -5.0490e-01,  6.6081e-07,
           1.9286e+00,  0.0000e+00])
        q_d = np.array([-3.72483757e-06,  1.23533666e+00,  7.29216661e-06, -1.33108648e-01,
        7.61155450e-07,  8.73456051e-01,  7.72071616e-20])
        s = np.linspace(0., 1., self._n_q_pts)
        #sh = np.linspace(0., 1., self.horizon)
        sh = np.linspace(0., 1., self.N.shape[1])
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

        self._add_save_attr(
            q_bias='pickle',
        )

    def generate_basis(self):
        phi = BasisFeatures(GaussianRBF.generate([self._n_q_pts], [0.], [1.], eta=0.95))
        dphi = BasisFeatures(dGaussianRBF.generate([self._n_q_pts], [0.], [1.], eta=0.95))
        self.N = np.stack([phi(i) for i in np.linspace(0, 1, self.horizon)], axis=0)[None]
        self.dN = np.stack([dphi(i) for i in np.linspace(0, 1, self.horizon)], axis=0)[None]

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
    
    def compute_trajectory_from_theta(self, theta, context):
        q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, puck = self.unpack_context(context)

        trainable_q_cps = theta[..., :-1].reshape(-1, self._n_trainable_q_pts, self.n_dim)
        trainable_t_scale = theta[..., -1:].reshape(-1)
        trainable_q_cps = trainable_q_cps * self.q_scale
        trainable_t_scale = trainable_t_scale * self.t_scale
        trainable_t_scale = torch.exp(trainable_t_scale)
        trainable_q = torch.tanh(trainable_q_cps) * np.pi

        N0 = torch.tensor(self.N[:, 0])
        q_cps_n0 = trainable_q + self.q_bias[None, 1:]
        #q_cps_n0 = self.q_bias[None, 1:]

        q_cps_0 = (q_0 - N0[:, 1:] @ q_cps_n0) / N0[:, 0]
        q_cps = torch.cat([q_cps_0, q_cps_n0], axis=-2)

        #q = self.N @ q_cps.detach().numpy()
        #for i in range(6):
        #    plt.subplot(231+i)
        #    plt.plot(q[0, :, i])
        #    plt.plot([0], q_0[0, :, i], 'gx')
        #    #plt.plot([150], q_d[0, :, i], 'rx')
        #plt.show()

        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps, trainable_t_scale, differentiable=True)
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

    def compute_trajectory(self, q_cps, t_scale, differentiable=False):
        N = self.N
        dN = self.dN
        if differentiable:
            N = torch.tensor(N)
            dN = torch.tensor(dN)

        q = N @ q_cps
        q_dot = dN @ q_cps
        q_ddot = torch.zeros_like(q_dot)

        q_dot /= t_scale[:, None, None]
        q_ddot /= t_scale[:, None, None]**2

        duration = t_scale
        s = torch.linspace(0., 1., N.shape[1])[None, :]
        t = (1 - s) * torch.zeros_like(duration[:, None]) + s * duration[:, None]
        dt = (duration / N.shape[1])[:, None].repeat(1, N.shape[1])

        return q, q_dot, q_ddot, t, dt, duration
