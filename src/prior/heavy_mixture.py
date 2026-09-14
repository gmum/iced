import random

from src.configs import DistributionType
from src.prior.augmentations import augment_data
from src.prior.scaler import MinMaxScalerTorch
from dataclasses import dataclass
from typing import Tuple, List

import torch
from torch import Tensor
import numpy as np
import math


@dataclass
class HeavyTailedMixture:
    probs: Tensor
    A: List[Tensor]
    b: List[Tensor]
    scaler: "MinMaxScalerTorch"
    dist: DistributionType = DistributionType.STUDENT_T
    df: float = 3.0

    def _log_base_density(self, z: Tensor) -> Tensor:
        if self.dist == DistributionType.STUDENT_T:
            df_t = torch.tensor(self.df, device=z.device)
            c = (
                torch.lgamma((df_t + 1) / 2)
                - torch.lgamma(df_t / 2)
                - 0.5 * torch.log(df_t * torch.pi)
            )
            log_pdf = c - ((df_t + 1) / 2) * torch.log(1 + (z ** 2) / df_t)

        elif self.dist == DistributionType.LAPLACE:
            log_pdf = -torch.abs(z) - math.log(2)

        elif self.dist == DistributionType.CAUCHY:
            pi = torch.tensor(math.pi, device=z.device)
            log_pdf = -torch.log(pi) - torch.log(1 + z ** 2)

        else:
            raise ValueError(f"Unsupported distribution: {self.dist}")

        return log_pdf.sum(dim=-1)

    def log_prob(self, x: Tensor, rescale: bool = False) -> Tensor:
        if rescale:
            x = self.scaler.inverse_transform(x)

        log_probs = []

        for k in range(len(self.A)):
            A_k = self.A[k]
            b_k = self.b[k]

            A_inv = torch.linalg.inv(A_k)
            sign, logdet = torch.linalg.slogdet(A_k)

            #if sign <= 0:
            #    raise ValueError("A_k must have positive determinant")

            z = (x - b_k) @ A_inv.T
            log_base = self._log_base_density(z)

            log_pk = log_base - logdet
            log_probs.append(log_pk + torch.log(self.probs[k]))

        log_probs = torch.stack(log_probs, dim=1)
        return torch.logsumexp(log_probs, dim=1)

    def transform(
        self,
        X_support: Tensor,
        X_query: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        log_probs_support = self.log_prob(X_support, rescale=True)
        log_probs_query = self.log_prob(X_query, rescale=True)
        return X_support, X_query, log_probs_support, log_probs_query


# =========================
# BASE SAMPLING
# =========================
def sample_base(dist: DistributionType, size, df=3.):
    if dist == DistributionType.STUDENT_T:
        return np.random.standard_t(df=df, size=size)
    elif dist == DistributionType.LAPLACE:
        return np.random.laplace(size=size)
    elif dist == DistributionType.CAUCHY:
        return np.random.standard_cauchy(size=size)
    else:
        raise ValueError("Unsupported distribution")


def create_heavy_tailed_mixture(
    num_gaussians, cluster_min_points, cluster_max_points,
    dim, n_support_min, n_support_max,
    n_query, add_noise, noise_std, augment_type,
    mean_range, dist: DistributionType = DistributionType.STUDENT_T,
    df: float = 3.0,
):
    means, A_list = [], []
    points = []

    dataset_components = np.random.randint(2, num_gaussians + 1)
    cur_dim = np.random.randint(2, dim + 1)

    for _ in range(dataset_components):
        points_per_gaussian = np.random.randint(
            cluster_min_points, cluster_max_points + 1
        )

        # TODO ensure that A is invertible in better way
        A = np.random.randn(cur_dim, cur_dim)
        A = A @ A.T
        while np.linalg.det(A) == 0:
            A = np.random.randn(cur_dim, cur_dim)
            A = A @ A.T

        b = np.random.uniform(-mean_range, mean_range, size=(cur_dim,))

        points.append(points_per_gaussian)
        A_list.append(torch.tensor(A, dtype=torch.float32))
        means.append(torch.tensor(b, dtype=torch.float32))

    points = torch.tensor(points, dtype=torch.float32)
    probs = points / torch.sum(points)

    n_support = np.random.randint(n_support_min, n_support_max)
    support_data, query_data = [], []

    for k in range(dataset_components):
        prob = probs[k].item()

        gauss_support = int(n_support * prob)
        gauss_query = int(n_query * prob)
        total = gauss_support + gauss_query

        # --- sample ---
        z = sample_base(dist, (total, cur_dim), df=df)

        A = A_list[k].numpy()
        b = means[k].numpy()

        x = (A @ z.T).T + b

        support_data.append(x[:gauss_support])
        query_data.append(x[gauss_support:])

    X_support = torch.tensor(np.vstack(support_data), dtype=torch.float32)
    X_query = torch.tensor(np.vstack(query_data), dtype=torch.float32)

    add_now = random.random() < 0.5
    if add_noise and add_now:
        X_query = augment_data(X_support, X_query, augment_type)

    scaler = MinMaxScalerTorch(feature_range=(-1, 1))
    X_support = scaler.fit_transform(X_support)
    X_query = scaler.transform(X_query)

    mixture = HeavyTailedMixture(
        probs=probs,
        A=A_list,
        b=means,
        scaler=scaler,
        dist=dist,
        df=df,
    )

    return X_support, X_query, cur_dim, mixture
