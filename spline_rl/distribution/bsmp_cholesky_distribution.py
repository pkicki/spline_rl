import torch
import numpy as np

from mushroom_rl.distributions import AbstractGaussianTorchDistribution
from spline_rl.utils.utils import project_entropy_chol, project_entropy_independently

class CholeskyDiagonalGaussianBSMPSigmaDistribution(AbstractGaussianTorchDistribution):
    def __init__(self, mu_approximator, chol_sigma_approximator, e_lb=None):
        self._mu_approximator = mu_approximator
        self._chol_sigma_approximator = chol_sigma_approximator
        self._e_lb = e_lb

        super().__init__(context_shape=self._mu_approximator.input_shape)

        self._add_save_attr(
            _mu_approximator='torch',
            _chol_sigma_approximator='torch',
            _e_lb='primitive',
        )

    def set_e_lb(self, e_lb):
        self._e_lb = e_lb

    def parameters(self):
        return list(self._mu_approximator.model.network.parameters()) + list(self._chol_sigma_approximator.model.network.parameters())

    def estimate_mu(self, context):
        if context is None:
            context = np.zeros(self._mu_approximator.input_shape, dtype=np.float32)#[None]
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context)
        mu = self._mu_approximator(context)
        return mu

    def estimate_chol_sigma(self, context):
        if context is None:
            context = np.zeros(self._chol_sigma_approximator.input_shape, dtype=np.float32)#[None]
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context)
        chol_sigma = self._chol_sigma_approximator(context)
        return chol_sigma

    def _get_mean_and_chol(self, context):
        mu = self.estimate_mu(context)
        chol = self.estimate_chol_sigma(context)
        if self._e_lb is not None:
            chol = project_entropy_chol(chol, self._e_lb)
        return mu, chol
