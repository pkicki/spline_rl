from spline_rl.policy.mp_policy import MPPolicy
import torch
import numpy as np
from scipy.interpolate import interp1d

from mp_pytorch.mp import ExpDecayPhaseGenerator, ProDMPBasisGenerator
from mushroom_rl.policy import Policy
from baseline.baseline_agent.optimizer import TrajectoryOptimizer
from spline_rl.utils.utils import unpack_data_airhockey
from spline_rl.policy.promp_policy import ProMPPolicy

import matplotlib.pyplot as plt


class ProDMPPolicy(MPPolicy):

    def generate_basis(self):
        phase_gn = ExpDecayPhaseGenerator()
        basis_gn = ProDMPBasisGenerator(
            phase_generator=phase_gn,
            num_basis=self._n_trainable_q_pts - 1,
            dt=1./1024)
        t = torch.linspace(0., 1., 1024).to(torch.float32)
        b = basis_gn.basis(t)
        db = basis_gn.vel_basis(t)

        self.N = b.detach().numpy()[None].astype(np.float64)
        self.dN = db.detach().numpy()[None].astype(np.float64)

        #for i in range(self.N.shape[-1] - 1):
        #    plt.plot(self.N[0, :, i], label=f'{i}')
        #plt.legend()
        #plt.show()
        a = 0

    def load_policy(self, env_info):
        self.optimizer = TrajectoryOptimizer(env_info)

    def compute_trajectory_from_theta(self, theta, context):
        q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, puck = self.unpack_context(context)

        #theta = torch.randn(100, 71)
        trainable_q_cps = theta[..., :-1].reshape(-1, self._n_trainable_q_pts, self.n_dim)
        trainable_t_scale = theta[..., -1:].reshape(-1)
        trainable_q_cps = trainable_q_cps * self.q_scale
        trainable_t_scale = trainable_t_scale * self.t_scale
        trainable_t_scale = torch.exp(trainable_t_scale)
        #middle_trainable_q_pts = torch.tanh(1000. * trainable_q_cps[:, :-1]) * np.pi
        middle_trainable_q_pts = 1000. * trainable_q_cps[:, :-1]
        trainable_q_d = torch.tanh(trainable_q_cps[:, -1:]) * np.pi

        x_des = np.array([1.31, 0., self.desired_ee_z])
        _, q_d_bias = self.optimizer.inverse_kinematics(x_des, q_0.detach().numpy()[0, 0])

        q_d = trainable_q_d + torch.tensor(q_d_bias)[None, None] - q_0

        q_cps = torch.cat([middle_trainable_q_pts, q_d], axis=-2)

        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_0, q_cps, trainable_t_scale, differentiable=True)

        #for i in range(self.n_dim):
        #    plt.subplot(3, 3, 1 + i)
        #    for k in range(t.shape[0]):
        #        plt.plot(t.detach().numpy()[k], q.detach().numpy()[k, :, i])
        #plt.show()

        self._traj_no += 1
        return q, q_dot, q_ddot, t, dt, duration

    def compute_trajectory(self, q_0, q_cps, t_scale, differentiable=False):
        N = self.N
        dN = self.dN
        if differentiable:
            N = torch.tensor(N)
            dN = torch.tensor(dN)

        q = N @ q_cps + q_0
        q_dot = dN @ q_cps
        q_ddot = torch.zeros_like(q_dot)

        q_dot /= t_scale[:, None, None]
        q_ddot /= t_scale[:, None, None]**2

        duration = t_scale
        s = torch.linspace(0., 1., N.shape[1])[None, :]
        t = (1 - s) * torch.zeros_like(duration[:, None]) + s * duration[:, None]
        dt = (duration / N.shape[1])[:, None].repeat(1, N.shape[1])

        return q, q_dot, q_ddot, t, dt, duration
