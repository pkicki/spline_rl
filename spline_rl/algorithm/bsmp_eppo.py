import os
import torch
import numpy as np

from mushroom_rl.algorithms.policy_search import ePPO
from mushroom_rl.utils.minibatches import minibatch_generator
from storm_kit.differentiable_robot_model import DifferentiableRobotModel


class BSMPePPO(ePPO):
    """
    Episodic adaptation of the Proximal Policy Optimization algorithm
    with B-spline actions and differentiable constraints.
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al.. 2017.
    "Fast Kinodynamic Planning on the Constraint Manifold With Deep Neural Networks"
    Kicki P. et al. 2023.
    """

    def __init__(self, mdp_info, distribution, policy, optimizer, value_function, value_function_optimizer,
                 constraint_lr, n_epochs_policy, batch_size, eps_ppo, target_entropy,
                 entropy_lr, initial_entropy_bonus, ent_coeff=0.0, context_builder=None): 
        self.alphas = np.array([0.] * 18)
        self.violation_limits = np.array([1e-4] * 7 + [1e-5] * 7 + [5e-6] * 4)
        self.constraint_lr = constraint_lr
        self.constraint_losses = []
        self.scaled_constraint_losses = []
        self.prev_scaled_constraint_losses = None
        self.task_losses = []
        self.prev_task_losses = None
        self.constraint_losses_log = []

        self.prev_delta = 0.
        self.delta_integral = 0.

        self.value_function = value_function
        self.value_function_optimizer = value_function_optimizer

        self._epoch_no = 0

        self._q = None
        self._q_dot = None
        self._q_ddot = None
        self._t = None
        self._ee_pos = None

        self.constraints = mdp_info.constraints

        super().__init__(mdp_info, distribution, policy, optimizer, n_epochs_policy,
                         batch_size, eps_ppo, ent_coeff, context_builder)
        
        self._log_entropy_bonus = torch.tensor(np.log(initial_entropy_bonus), dtype=torch.float32, requires_grad=True)
        self._entropy_optimizer = torch.optim.Adam([self._log_entropy_bonus], lr=entropy_lr)
        self._target_entropy = target_entropy

        self._add_save_attr(
            alphas='numpy',
            violation_limits='numpy',
            constraint_lr='primitive',
            sigma_optimizer='torch',
            mu_approximator='mushroom',
            mu_optimizer='torch',
            value_function='torch',
            value_function_optimizer='torch',
            constraint_losses='pickle',
            constraint_losses_log='pickle',
            _log_entropy_bonus='torch',
            _entropy_optimizer='torch',
            _target_entropy='primitive',
            _epoch_no='primitive',
        )

    def episode_start(self, initial_state, episode_info):
        _, theta = super().episode_start(initial_state, episode_info)
        return self._convert_to_env_backend(self.policy.reset(initial_state)), theta

    def _unpack_qt(self, qt, trainable=False):
        n_q_pts = self._n_trainable_q_pts if trainable else self._n_q_pts
        q = qt[..., :self._n_dim * n_q_pts]
        t = qt[..., self._n_dim * n_q_pts:]
        return q, t

    #def update_alphas_test(self):
    #    if self.prev_task_losses is None:
    #        self.prev_task_losses = self.task_losses.copy()
    #    if self.prev_scaled_constraint_losses is None:
    #        self.prev_scaled_constraint_losses = self.scaled_constraint_losses.copy()

    #    constraint_losses = torch.mean(torch.cat(self.constraint_losses, dim=0), dim=0)

    #    def compute_losses(task_losses, constraint_losses):
    #        task_losses = torch.mean(torch.cat(task_losses, dim=0)[:, None], dim=0)
    #        constraint_losses = torch.mean(torch.cat(constraint_losses, dim=0), dim=0)
    #        task_constraint_losses = torch.cat([task_losses, constraint_losses], dim=0)
    #        m = torch.ones((len(self.alphas), len(self.alphas) + 1))
    #        m[torch.arange(len(self.alphas)), torch.arange(len(self.alphas)) + 1] = 0.
    #        rest_losses = m @ task_constraint_losses
    #        return task_losses, constraint_losses, rest_losses
    #    
    #    prev_task_losses, prev_scaled_constraint_losses, prev_rest_losses = \
    #        compute_losses(self.prev_task_losses, self.prev_scaled_constraint_losses)
    #    task_losses, scaled_constraint_losses, rest_losses = \
    #        compute_losses(self.task_losses, self.scaled_constraint_losses)

    #    rest_losses_diff = rest_losses - prev_rest_losses

    #    alphas_update = self.constraint_lr * np.log(
    #        (constraint_losses.detach().numpy() + self.violation_limits * 1e-1) / self.violation_limits)
    #    self.alphas += alphas_update
    #    self.alphas = np.clip(self.alphas, -7., None)
    #    self.constraint_losses_log = constraint_losses
    #    self.constraint_losses = []
    #    self.prev_task_losses = self.task_losses.copy()
    #    self.prev_scaled_constraint_losses = self.scaled_constraint_losses.copy()

    def update_alphas(self):
        constraint_losses = np.mean(np.concatenate(self.constraint_losses, axis=0), axis=0)
        alphas_update = self.constraint_lr * np.log(
            (constraint_losses + self.violation_limits * 1e-1) / self.violation_limits)
        self.alphas += alphas_update
        self.alphas = np.clip(self.alphas, -7., None)
        self.constraint_losses_log = constraint_losses
        self.constraint_losses = []

    #def update_alphas_PID(self):
    #    constraint_losses = np.mean(np.concatenate(self.constraint_losses, axis=0), axis=0)
    #    delta = constraint_losses - self.violation_limits
    #    deriv = delta - self.prev_delta
    #    self.delta_integral += delta
    #    kp = 1.
    #    ki = 0.01
    #    kd = 0.1
    #    self.alphas = np.maximum(kp * delta + ki * self.delta_integral + kd * deriv, 0.)
    #    self.prev_delta = delta
    #    self.constraint_losses_log = constraint_losses
    #    self.constraint_losses = []

    def update_entropy_bonus(self, log_prob):
        entropy_loss = - (self._log_entropy_bonus.exp() * (log_prob + self._target_entropy)).mean()
        self._entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self._entropy_optimizer.step()

    # All constraint losses computation organized in a single function
    def compute_constraint_losses(self, theta, context):
        q, q_dot, q_ddot, t, dt, duration = self.policy.compute_trajectory_from_theta(theta, context)
        return self.constraints.evaluate(q, q_dot, q_ddot, dt)

    def _update(self, Jep, theta, context):
        if len(theta.shape) == 3:
            theta = theta[:, 0]

        Jep = torch.tensor(Jep)
        #J_mean = torch.mean(Jep)
        #J_std = torch.std(Jep)

        #Jep = (Jep - J_mean) / (J_std + 1e-8)

        with torch.no_grad():
            value = self.value_function(context)[:, 0]
            mean_advantage = torch.mean(Jep - value)

        old_dist = self.distribution.log_pdf(theta, context).detach()

        if self.distribution.is_contextual:
            full_batch = (theta, Jep, old_dist, context)
        else:
            full_batch = (theta, Jep, old_dist)

        for epoch in range(self._n_epochs_policy()):
            for minibatch in minibatch_generator(self._batch_size(), *full_batch):
                self._optimizer.zero_grad()
                theta_i, context_i, Jep_i, old_dist_i = self._unpack(minibatch)

                # ePPO loss
                lp = self.distribution.log_pdf(theta_i, context_i)
                prob_ratio = torch.exp(lp - old_dist_i)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(), 1 + self._eps_ppo.get_value())
                value_i = self.value_function(context_i)[:, 0]
                A = Jep_i - value_i
                A_unbiased = A - mean_advantage
                task_loss = -torch.min(prob_ratio * A_unbiased, clipped_ratio * A_unbiased)

                # constraint loss
                mu = self.distribution.estimate_mu(context_i)
                constraint_losses = self.compute_constraint_losses(mu, context_i)
                self.constraint_losses.append(constraint_losses.detach().numpy())
                #self.task_losses.append(task_loss.detach())
                constraint_loss = torch.exp(torch.Tensor(self.alphas))[None] * constraint_losses
                #self.scaled_constraint_losses.append(constraint_loss.detach())
                constraint_loss = torch.sum(constraint_loss, dim=-1)
                loss = torch.mean(task_loss) + torch.mean(constraint_loss)

                value_loss = torch.mean(A**2)
                loss.backward(retain_graph=True)
                self._optimizer.step()
                self.value_function_optimizer.zero_grad()
                value_loss.backward()
                self.value_function_optimizer.step()
            self.update_alphas()
            #self.update_alphas_PID()
            #self.update_entropy_bonus(self.distribution.log_pdf(theta, context))
            self._epoch_no += 1
            #mu = self.distribution._mu