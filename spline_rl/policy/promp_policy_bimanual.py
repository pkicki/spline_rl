import torch
import numpy as np
import matplotlib.pyplot as plt
from spline_rl.policy.promp_policy import ProMPPolicy

from spline_rl.utils.utils import unpack_data_bimanual



class ProMPPolicyBimanual(ProMPPolicy):
    def __init__(self, env_info, n_q_cps, n_dim, n_pts_fixed_begin=1,
                 t_scale=1., q_scale=1., q_d_scale=1., q_dot_d_scale=1., q_ddot_d_scale=1., **kwargs):
        super().__init__(env_info, n_q_cps, n_dim, n_pts_fixed_begin,
                         t_scale, q_scale, q_d_scale, q_dot_d_scale, q_ddot_d_scale)
        self.q_d_bias = torch.tensor([
            0.,
            0.21646051,  0.48360129,  0.55410618, -0.17025892, -0.00847046,  0.30138272,
            -0.04727616, -0.64639233, -0.52647715, 0.1963449 , -0.01348486, -0.06694195])

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
        theta = theta.to(torch.float64) if type(theta) is torch.Tensor else theta.astype(np.float64)
        context = context.to(torch.float64) if type(context) is torch.Tensor else context.astype(np.float64)
        q_0, q_dot_0, q_ddot_0, _, q_dot_d, q_ddot_d = self.unpack_context(context)

        trainable_q_cps = theta[..., :-1].reshape(-1, self._n_trainable_q_pts, self.n_dim)
        trainable_t_scale = theta[..., -1:].reshape(-1)


        ## unstructured
        #trainable_q_cps = trainable_q_cps * self.q_scale
        #trainable_t_scale = trainable_t_scale * self.t_scale
        #trainable_t_scale = torch.exp(trainable_t_scale)
        #trainable_q = torch.tanh(trainable_q_cps) * 2. * np.pi
        #N0 = torch.tensor(self.N[:, 0])
        ##q_cps_n0 = trainable_q + self.q_bias[None, 1:]
        #q_cps_n0 = trainable_q + q_0
        ##q_cps_n0 = self.q_bias[None, 1:]

        #q_cps_0 = (q_0 - N0[:, 1:] @ q_cps_n0) / N0[:, 0]
        #q_cps = torch.cat([q_cps_0, q_cps_n0], axis=-2)

        ## structured
        trainable_q_middle_cps = trainable_q_cps[:, :-1] * self.q_scale
        trainable_q_d = trainable_q_cps[:, -1:] * self.q_d_scale
        trainable_t_scale = trainable_t_scale * self.t_scale
        trainable_t_scale = torch.exp(trainable_t_scale)
        trainable_q_middle = torch.tanh(trainable_q_middle_cps) * 2. * np.pi
        trainable_q_d = torch.tanh(trainable_q_d) * 2 * np.pi
        s = torch.linspace(0., 1., trainable_q_middle_cps.shape[1]+2)[None, 1:-1, None]
        q_d = trainable_q_d + self.q_d_bias[None, None]
        q_b = q_0 * (1 - s) + q_d * s
        q_cps_middle = q_b + trainable_q_middle

        N0 = torch.tensor(self.N[:, 0])
        q_cps_0 = (q_0 - N0[:, 1:-1] @ q_cps_middle) / N0[:, 0]
        Nm1 = torch.tensor(self.N[:, -1])
        q_cps_d = (q_d - Nm1[:, 1:-1] @ q_cps_middle) / Nm1[:, -1]
        q_cps_d = q_cps_d + trainable_q_d
        q_cps = torch.cat([q_cps_0, q_cps_middle, q_cps_d], axis=-2)

        #q = self.N @ q_cps.detach().numpy()
        #for i in range(q.shape[-1]):
        #    plt.subplot(4, 4, 1+i)
        #    plt.plot(q[0, :, i])
        #    plt.plot([0], q_0[0, :, i], 'gx')
        #    plt.plot([q.shape[-2]], q_d[0, :, i], 'rx')
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