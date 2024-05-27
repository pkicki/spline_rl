import numpy as np
from spline_rl.policy.mp_policy import MPPolicy
from spline_rl.policy.promp_policy import ProMPPolicy
import torch
from scipy.interpolate import interp1d

from mushroom_rl.policy import Policy
from baseline.baseline_agent.optimizer import TrajectoryOptimizer

from mushroom_rl.features._implementations.basis_features import BasisFeatures
from mushroom_rl.features.basis import GaussianRBF
from spline_rl.utils.gaussian_derivative import dGaussianRBF
from spline_rl.utils.utils import unpack_data_airhockey


class ProMPPolicyKino(ProMPPolicy):
    def __init__(self, env_info, n_q_cps, n_dim, n_pts_fixed_begin=1,
                 t_scale=1., q_scale=1., q_d_scale=1., q_dot_d_scale=1., q_ddot_d_scale=1., **kwargs):
        super().__init__(env_info, n_q_cps, n_dim, n_pts_fixed_begin,
                         t_scale, q_scale, q_d_scale, q_dot_d_scale, q_ddot_d_scale)
        q_0 = np.array([0.3141, -0.6135, -0.8648,  1.5832, -1.3138, -1.0563,  0.4567])
        q_d = np.array([-0.1178,  0.2472, -2.1531,  2.0209, -2.6691, -0.7095,  0.9851])
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