from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor

from src.configs import Nonlinearity
from src.prior.gaussian_mixture import GaussianMixture


# -------------------------
# Utility: Random orthogonal matrix
# -------------------------
def random_orthogonal(d):
    """
    Generate a random orthogonal matrix Q of size d x d
    """
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    return Q


# -------------------------
# Flow Layer
# -------------------------
class FlowLayer(nn.Module):
    def __init__(self, d, slope_coeff: float = 0.5, nonlinearity: Nonlinearity = Nonlinearity.PIECEWISE):
        """
        d : int - input dimension
        nonlinearity : str - 'piecewise', 'elu', or 'softplus'
        """
        super().__init__()
        self.d = d
        self.Q = nn.Parameter(random_orthogonal(d), requires_grad=False)
        self.nonlinearity = nonlinearity

        # Piecewise Linear parameters
        if nonlinearity == Nonlinearity.PIECEWISE:
            self.a = slope_coeff + (1 - slope_coeff) * torch.rand(d)
            self.b = 1 + (1 / slope_coeff - 1) * torch.rand(d)
        # ELU parameters
        elif nonlinearity == Nonlinearity.ELU:
            self.alpha = slope_coeff + (1 - slope_coeff) * torch.rand(d)
            self.beta = 1 + (1 / slope_coeff - 1) * torch.rand(d)
        # Softplus parameters
        elif nonlinearity == Nonlinearity.SOFTPLUS:
            self.gamma = slope_coeff + (1 - slope_coeff) * torch.rand(d)
            self.s = 1 + (1 / slope_coeff - 1) * torch.rand(d)
            self.m = slope_coeff + (1 - slope_coeff) * torch.rand(d)

    def forward(self, z):
        """
        Forward pass through the flow layer
        z: input tensor of shape (batch, d)
        Returns transformed tensor and log-determinant
        """


        # 1. Linear orthogonal mixing
        z_prime = z @ self.Q.T  # Q z
        logdet_linear = 0.0  # |det(Q)| = 1

        mu = z_prime.mean(dim=0)
        std = z_prime.std(dim=0) + 1e-6

        alpha = 0.3  # concentration strength (smaller = tighter)

        eps = torch.randn_like(mu)
        self.k = mu + alpha * std * eps

        # 2. Nonlinear transformation
        if self.nonlinearity == Nonlinearity.PIECEWISE:
            left = self.a * (z_prime - self.k) + self.k
            right = self.b * (z_prime - self.k) + self.k
            mask = (z_prime <= self.k).float()
            z_out = mask * left + (1 - mask) * right
            logdet_nl = torch.sum(torch.log(mask * self.a + (1 - mask) * self.b), dim=1)

        elif self.nonlinearity == Nonlinearity.ELU:
            mask = (z_prime <= self.k).float()
            left = self.alpha * (torch.exp(z_prime - self.k) - 1) + self.k
            right = self.beta * (z_prime - self.k) + self.k
            z_out = mask * left + (1 - mask) * right
            grad_left = self.alpha * torch.exp(z_prime - self.k)
            grad_right = self.beta
            logdet_nl = torch.sum(torch.log(mask * grad_left + (1 - mask) * grad_right), dim=1)

        elif self.nonlinearity == Nonlinearity.SOFTPLUS:
            z_shift = z_prime - self.k
            z_out = self.s / self.gamma * torch.nn.functional.softplus(self.gamma * z_shift) + self.m * z_prime
            grad = self.s * torch.sigmoid(self.gamma * z_shift) + self.m
            logdet_nl = torch.sum(torch.log(grad), dim=1)

        else:
            raise ValueError("Unsupported nonlinearity")

        logdet = logdet_linear + logdet_nl
        return z_out, logdet


# -------------------------
# Full Flow Model
# -------------------------
class FlowModel(nn.Module):
    def __init__(self, gmm_data: GaussianMixture, features, L=3, slope_coeff=0.5, nonlinearity = Nonlinearity.PIECEWISE):
        super().__init__()
        self.layers = nn.ModuleList([FlowLayer(features, slope_coeff=slope_coeff, nonlinearity=nonlinearity) for _ in range(L)])
        self.gmm = gmm_data

    def forward(self, x):
        logdet_total = 0
        z = x
        for layer in self.layers:
            z, logdet = layer(z)
            logdet_total += logdet
        return z, logdet_total

    def transform(self, X_support: Tensor, X_query: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        prior_log_support = self.gmm.log_prob(X_support, rescale=True)
        prior_log_query = self.gmm.log_prob(X_query, rescale=True)

        # Transform support
        X_support_trans, log_det_support = self(X_support)
        # Transform query
        X_query_trans, log_det_query = self(X_query)

        # Apply MinMax scaling to [-1, 1]
        min_scaler = MinMaxScaler(feature_range=(-1, 1))
        X_support_scaled = torch.tensor(min_scaler.fit_transform(X_support_trans), dtype=X_support_trans.dtype)
        X_query_scaled = torch.tensor(min_scaler.transform(X_query_trans), dtype=X_query_trans.dtype)
        self.scaler = min_scaler

        log_probs_support = prior_log_support - log_det_support
        log_probs_query = prior_log_query - log_det_query

        return X_support_scaled, X_query_scaled, log_probs_support, log_probs_query
