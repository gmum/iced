import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from src.configs import AugmentationType
from src.prior.augmentations import augment_data
from src.prior.scaler import MinMaxScalerTorch


@dataclass
class GaussianMixture:
    probs: Tensor
    means: Tensor
    covs: Tensor
    scaler: MinMaxScalerTorch

    def log_prob(self, x: Tensor, rescale: bool = False) -> Tensor:
        """
        Numerically stable batched log-density for a GMM with full covariances.
        x: (N, D)
        returns: (N,) log p(x)
        Assumes:
          - self.probs: (K,) (sums to 1)
          - self.means: (K, D)
          - self.covs: (K, D, D) (positive-definite)
        """
        if rescale:
            x = self.scaler.inverse_transform(x)

        N, D = x.shape

        # (N, K, D)
        diff = x[:, None, :] - self.means[None, :, :]
        # (K, N, D)
        diff_knd = diff.permute(1, 0, 2)

        # Cholesky: cov_k = L_k @ L_k^T, L lower triangular, shape (K, D, D)
        L = torch.linalg.cholesky(self.covs)  # will error if a cov isn't PD

        # Prepare RHS for solve: (K, D, N)
        B = diff_knd.transpose(1, 2)

        # Solve L @ y = B  -> y has shape (K, D, N)
        y = torch.linalg.solve_triangular(L, B, upper=False)

        # Mahalanobis per (k,n): sum_d y^2
        mahal_k_n = (y ** 2).sum(dim=1)  # (K, N)
        mahal = mahal_k_n.transpose(0, 1)  # (N, K)

        # logdet(cov_k) = 2 * sum(log(diag(L_k)))
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1)  # (K,)

        const = D * math.log(2 * math.pi)
        log_gauss = -0.5 * (mahal + log_det[None, :] + const)  # (N, K)

        return torch.logsumexp(log_gauss + torch.log(self.probs)[None, :], dim=1)
    
    def transform(self, X_support: Tensor, X_query: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        log_probs_support = self.log_prob(X_support, rescale=True)
        log_probs_query = self.log_prob(X_query, rescale=True)

        return X_support, X_query, log_probs_support, log_probs_query


def create_gaussian_mixture(
        num_gaussians, cluster_min_points, cluster_max_points, dim, n_support_min,
        n_support_max, n_query, add_noise, noise_std, augment_type, mean_range
):
    means, covs = [], []
    points = []

    dataset_gaussians = np.random.randint(2, num_gaussians + 1)

    # Ensure reasonable number of points per cluster
    #cluster_max_points = min(max_points, 2500 // dataset_gaussians + 1)
    cur_dim = np.random.randint(2, dim + 1)

    for gaussian_index in range(dataset_gaussians):
        points_per_gaussian = np.random.randint(cluster_min_points, cluster_max_points + 1)

        random_matrix = np.random.randn(cur_dim, cur_dim)
        covariance_matrix = np.dot(random_matrix, random_matrix.T)
        cov = covariance_matrix

        mean = np.random.uniform(-mean_range, mean_range, size=(cur_dim, ))

        points.append(points_per_gaussian)

        means.append(mean[np.newaxis, :])
        covs.append(cov[np.newaxis, :])

    points = torch.tensor(points, dtype=torch.float32)
    # More stable probability calculation
    probs = points / torch.sum(points)

    n_support = np.random.randint(n_support_min, n_support_max)
    support_data, query_data = [], []

    for cur_mean, cur_cov, prob in zip(means, covs, probs):
        gauss_support = int(n_support*prob)
        gauss_query = int(n_query*prob)
        points_per_gaussian = gauss_support + gauss_query

        gaussian_data = np.random.multivariate_normal(cur_mean[0], cur_cov[0], points_per_gaussian)

        # Combine continuous and categorical features
        combined_data = gaussian_data
        x_support = combined_data[:gauss_support]
        x_query = combined_data[gauss_support:]

        support_data.append(x_support)
        query_data.append(x_query)

    X_support = np.vstack(support_data)
    X_query = np.vstack(query_data)

    means = torch.from_numpy(np.concatenate(means, axis=0))
    covs = torch.from_numpy(np.concatenate(covs, axis=0))

    X_support = torch.tensor(X_support, dtype=torch.float32)
    X_query = torch.tensor(X_query, dtype=torch.float32)

    add_now = random.random() < 0.5
    if add_noise and add_now:
        X_query = augment_data(X_support, X_query, augment_type)

    min_scaler = MinMaxScalerTorch(feature_range=(-1, 1))
    X_support = min_scaler.fit_transform(X_support)
    X_query = min_scaler.transform(X_query)

    gmm_data = GaussianMixture(probs, means, covs, min_scaler)

    return X_support, X_query, cur_dim, gmm_data
