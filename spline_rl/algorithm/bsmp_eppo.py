import os
import torch
import numpy as np

from mushroom_rl.algorithms.policy_search import ePPO
from mushroom_rl.utils.minibatches import minibatch_generator



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
                 constraint_lr, n_epochs_policy, batch_size, eps_ppo, ent_coeff=0., kl_threshold=1e10, context_builder=None): 
        self.alphas = np.array([0.] * mdp_info.constraints.constraints_num)
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
        self.kl_threshold = kl_threshold
        self.last_kl_divergence = 0.

        self._q = None
        self._q_dot = None
        self._q_ddot = None
        self._t = None
        self._ee_pos = None

        self.load_constraints(mdp_info)

        super().__init__(mdp_info, distribution, policy, optimizer, n_epochs_policy,
                         batch_size, eps_ppo, ent_coeff, context_builder)
        
        self._add_save_attr(
            alphas='numpy',
            constraint_lr='primitive',
            sigma_optimizer='torch',
            mu_approximator='mushroom',
            mu_optimizer='torch',
            value_function='torch',
            value_function_optimizer='torch',
            constraint_losses='pickle',
            constraint_losses_log='pickle',
            kl_threshold='primitive',
            last_kl_divergence='primitive',
            _epoch_no='primitive',
        )

    def load_constraints(self, mdp_info):
        self.constraints = mdp_info.constraints

    def episode_start(self, initial_state, episode_info):
        _, theta = super().episode_start(initial_state, episode_info)
        return self._convert_to_env_backend(self.policy.reset(initial_state)), theta

    def draw_action(self, state, policy_state=None):
        action, policy_state = super().draw_action(state, policy_state)
        if self.mdp_info.interpolation_order in [1, 2]:
            action = action[:1]
        elif self.mdp_info.interpolation_order in [-1, 3, 4]:
            action = action[:2]
        return action, policy_state

    def _unpack_qt(self, qt, trainable=False):
        n_q_pts = self._n_trainable_q_pts if trainable else self._n_q_pts
        q = qt[..., :self._n_dim * n_q_pts]
        t = qt[..., self._n_dim * n_q_pts:]
        return q, t

    def update_alphas(self):
        constraint_losses = np.mean(np.concatenate(self.constraint_losses, axis=0), axis=0)
        alphas_update = self.constraint_lr * np.log(
            (constraint_losses + self.constraints.violation_limits * 1e-1) / self.constraints.violation_limits)
        self.alphas += alphas_update
        self.alphas = np.clip(self.alphas, -7., 37.)
        self.constraint_losses_log = constraint_losses
        self.constraint_losses = []

    def get_alphas(self):
        return self.alphas

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

        old_dist_ = self.distribution.distribution(context)
        old_lp = old_dist_.log_prob(theta).detach()
        #old_dist = self.distribution.log_pdf(theta, context).detach()
        #old_lp_ = self.distribution.log_pdf(theta, context).detach()

        if self.distribution.is_contextual:
            full_batch = (theta, Jep, old_lp, context)
        else:
            full_batch = (theta, Jep, old_lp)

        for epoch in range(self._n_epochs_policy()):
            self._optimizer.zero_grad()
            # ePPO loss
            new_dist_i = self.distribution.distribution(context)
            lp_i = new_dist_i.log_prob(theta)
            #lp_i = self.distribution.log_pdf(theta_i, context_i)
            log_ratio = lp_i - old_lp
            prob_ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(), 1 + self._eps_ppo.get_value())
            value_i = self.value_function(context)[:, 0]
            A = Jep - value_i
            A_unbiased = A - mean_advantage
            A_unbiased = A_unbiased.detach()
            task_loss = -torch.min(prob_ratio * A_unbiased, clipped_ratio * A_unbiased)

            with torch.no_grad():
                approx_kl_div_ = torch.distributions.kl.kl_divergence(old_dist_, new_dist_i).mean().cpu().numpy()
                #approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                self.last_kl_divergence = approx_kl_div_
            if approx_kl_div_ > self.kl_threshold:
                break

            # constraint loss
            #mu = self.distribution.estimate_mu(context_i)
            mu = new_dist_i.mean
            constraint_losses = self.compute_constraint_losses(mu, context)
            self.constraint_losses.append(constraint_losses.detach().numpy())
            constraint_loss = torch.exp(torch.Tensor(self.alphas))[None] * constraint_losses
            constraint_loss = torch.sum(constraint_loss, dim=-1)
            loss = torch.mean(task_loss) + torch.mean(constraint_loss)

            value_loss = torch.mean(A**2)
            loss.backward()
            #print("UPDATE EPOCH:", epoch)
            #print("SIGMA GRAD MAX:", self.distribution._log_sigma_approximator.model.network.fc[4].weight.grad.abs().max())
            #print("MU GRAD MAX:", self.distribution._mu_approximator.model.network.fc[4].weight.grad.abs().max())
            self._optimizer.step()
            self.value_function_optimizer.zero_grad()
            value_loss.backward()
            self.value_function_optimizer.step()
            self.update_alphas()
            self._epoch_no += 1
            #for minibatch in minibatch_generator(self._batch_size(), *full_batch):
            #    self._optimizer.zero_grad()
            #    theta_i, context_i, Jep_i, old_lp_i = self._unpack(minibatch)

            #    # ePPO loss
            #    new_dist_i = self.distribution.distribution(context_i)
            #    lp_i = new_dist_i.log_prob(theta_i)
            #    #lp_i = self.distribution.log_pdf(theta_i, context_i)
            #    log_ratio = lp_i - old_lp_i
            #    prob_ratio = torch.exp(log_ratio)
            #    clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(), 1 + self._eps_ppo.get_value())
            #    value_i = self.value_function(context_i)[:, 0]
            #    A = Jep_i - value_i
            #    A_unbiased = A - mean_advantage
            #    A_unbiased = A_unbiased.detach()
            #    task_loss = -torch.min(prob_ratio * A_unbiased, clipped_ratio * A_unbiased)

            #    # constraint loss
            #    #mu = self.distribution.estimate_mu(context_i)
            #    mu = new_dist_i.mean
            #    constraint_losses = self.compute_constraint_losses(mu, context_i)
            #    self.constraint_losses.append(constraint_losses.detach().numpy())
            #    constraint_loss = torch.exp(torch.Tensor(self.alphas))[None] * constraint_losses
            #    constraint_loss = torch.sum(constraint_loss, dim=-1)
            #    loss = torch.mean(task_loss) + torch.mean(constraint_loss)

            #    value_loss = torch.mean(A**2)
            #    loss.backward()
            #    print("UPDATE EPOCH:", epoch)
            #    print("SIGMA GRAD MAX:", self.distribution._log_sigma_approximator.model.network.fc[4].weight.grad.abs().max())
            #    print("MU GRAD MAX:", self.distribution._mu_approximator.model.network.fc[4].weight.grad.abs().max())
            #    self._optimizer.step()
            #    self.value_function_optimizer.zero_grad()
            #    value_loss.backward()
            #    self.value_function_optimizer.step()
            #self.update_alphas()
            #self._epoch_no += 1