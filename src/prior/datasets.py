import random
from itertools import islice

import numpy as np
import torch

from src.configs import NetworkType, AugmentationType, AugmentationMOMENT, Nonlinearity, MixtureType, DistributionType
from src.flows.nvp import SimpleRealNVP
from src.flows.special_flow import FlowModel
from src.prior.categorical import build_safe_class_counts, sample_dirichlet_categorical
from src.prior.gaussian_mixture import create_gaussian_mixture
from src.prior.heavy_mixture import create_heavy_tailed_mixture
import torch.nn.functional as F


def dataset_generator(
        num_gaussians, cluster_min_points, cluster_max_points, dim,
        n_support_min, n_support_max, n_query, estimate, n_series,
        n_hutch, network_type, noise_std, augment_type, augment_moment, mean_range,
        mixture_type, slope_coeff, num_layers, prob_cat
):

    while True:
        if augment_moment == AugmentationMOMENT.RANDOM:
            cur_augment_moment = random.choice([AugmentationMOMENT.PRE_TRANSFORM, AugmentationMOMENT.POST_TRANSFORM])
        else:
            cur_augment_moment = augment_moment
        add_noise = cur_augment_moment == AugmentationMOMENT.PRE_TRANSFORM

        try:
            cur_mixture_type = random.choice([MixtureType.GAUSSIAN, MixtureType.HEAVY_TAIL]) \
                if mixture_type == MixtureType.RANDOM else mixture_type

            if cur_mixture_type == MixtureType.GAUSSIAN:
                X_support, X_query, cur_dim, data_object = create_gaussian_mixture(
                    num_gaussians, cluster_min_points, cluster_max_points, dim, n_support_min,
                    n_support_max, n_query, add_noise, noise_std, augment_type, mean_range
                )
            else:
                dist = random.choice([DistributionType.LAPLACE, DistributionType.STUDENT_T, DistributionType.CAUCHY])
                X_support, X_query, cur_dim, data_object = create_heavy_tailed_mixture(
                    num_gaussians, cluster_min_points, cluster_max_points, dim, n_support_min,
                    n_support_max, n_query, add_noise, noise_std, augment_type, mean_range, dist=dist
                )

            if network_type == NetworkType.RANDOM:
                if cur_mixture_type == MixtureType.GAUSSIAN:
                    cur_network_type = random.choices(
                        [NetworkType.NONE, NetworkType.SPEC_FLOW, NetworkType.NVP],
                        weights=[0.8, 0.1, 0.1],
                        k=1
                    )[0]
                else:
                    cur_network_type = random.choices(
                        [NetworkType.NONE, NetworkType.SPEC_FLOW, NetworkType.NVP],
                        weights=[0.9, 0.0, 0.1],
                        k=1
                    )[0]

            else:
                cur_network_type = network_type

            if cur_network_type == NetworkType.NVP:
                net = SimpleRealNVP(data_object, X_support.shape[1], 64, 30, 2)
            elif cur_network_type == NetworkType.SPEC_FLOW:
                nonlinearity = random.choice([Nonlinearity.PIECEWISE, Nonlinearity.ELU, Nonlinearity.SOFTPLUS])

                t = (cur_dim - 2) / (dim - 2) if dim > 2 else 0
                gamma = 1
                t_nl = t ** gamma
                slope_coeff = slope_coeff + t_nl * (0.95 - slope_coeff)

                net = FlowModel(data_object, X_support.shape[1], L=num_layers,
                                nonlinearity=nonlinearity, slope_coeff=slope_coeff)
            else: # NetworkType.NONE
                net = data_object

            X_support_net, X_query_net, log_probs_support, log_probs = net.transform(
                X_support, X_query, estimate=estimate,
                n_series=n_series, n_hutch=n_hutch,
            )

            if np.random.random() < prob_cat:
                cur_num_categorical = np.random.randint(1, 6)
                class_counts = build_safe_class_counts(
                    d_num=X_support.shape[1],
                    d_cat=cur_num_categorical,
                    max_dim=50
                )

                if class_counts != []:
                    X_support_cat, X_query_cat, log_probs_support_cat, log_probs_cat = sample_dirichlet_categorical(
                        class_counts=class_counts,
                        n_support=X_support_net.shape[0],
                        n_query=X_query.shape[0],
                        alpha=1.0,
                        device=X_support_net.device
                    )

                    support_blocks = []
                    query_blocks = []

                    for i in range(X_support_net.shape[1]):
                        support_blocks.append(X_support_net[:, i:i + 1])
                        query_blocks.append(X_query_net[:, i:i + 1])

                    for j, Cj in enumerate(class_counts):
                        support_blocks.append(
                            F.one_hot(X_support_cat[:, j], Cj).float()
                        )
                        query_blocks.append(
                            F.one_hot(X_query_cat[:, j], Cj).float()
                        )

                    perm = torch.randperm(len(support_blocks))

                    X_support_net = torch.cat(
                        [support_blocks[i] for i in perm],
                        dim=1,
                    )

                    X_query_net = torch.cat(
                        [query_blocks[i] for i in perm],
                        dim=1,
                    )

                    log_probs += log_probs_cat
                    log_probs_support += log_probs_support_cat

            q_min = torch.quantile(log_probs_support, 0.1).item()
            q_max = torch.quantile(log_probs_support, 0.9).item()

            eps = 1e-8
            log_probs = 2.0 * (log_probs - q_min) / (q_max - q_min + eps) - 1.0

        except Exception as e:
            print("Error: ", e)
            continue

        if torch.isnan(X_support_net).any() or torch.isnan(X_query_net).any():
            continue

        yield X_support_net.detach().float(), X_query_net.detach().float(), log_probs.detach().float()


class GMMDataset(torch.utils.data.Dataset):
    def __init__(self, num_datasets, num_gaussians, cluster_min_points, cluster_max_points,
                 dim,n_support_min, n_support_max, n_query, estimate, n_series,
                 n_hutch, network_type, noise_std, augment_type, augment_moment,
                 mean_range, mixture_type, slope_coeff, num_layers, prob_cat):
        self.num_datasets = num_datasets
        self.num_gaussians = num_gaussians
        self.cluster_min_points = cluster_min_points
        self.cluster_max_points = cluster_max_points
        self.dim = dim
        self.n_support_min = n_support_min
        self.n_support_max = n_support_max
        self.n_query = n_query
        self.estimate = estimate
        self.n_series = n_series
        self.n_hutch = n_hutch
        self.network_type = network_type
        self.noise_std = noise_std
        self.augment_type = augment_type
        self.augment_moment = augment_moment
        self.mean_range = mean_range
        self.mixture_type = mixture_type
        self.slope_coeff = slope_coeff
        self.num_layers = num_layers
        self.prob_cat = prob_cat

        # Set random seed based on dataset index to ensure reproducibility
        self.base_seed = np.random.randint(0, 10000)

    def __len__(self):
        return self.num_datasets

    def __getitem__(self, idx):
        # Set seed for this specific dataset

        return list(islice(
            dataset_generator(
                self.num_gaussians, self.cluster_min_points, self.cluster_max_points, self.dim,
                self.n_support_min, self.n_support_max, self.n_query, self.estimate, self.n_series,
                self.n_hutch, self.network_type, self.noise_std, self.augment_type, self.augment_moment,
                self.mean_range, self.mixture_type, self.slope_coeff, self.num_layers, self.prob_cat),
            1)
        )[0]


def generate_gmm_datasets_with_projected_points(
        num_datasets, num_gaussians, cluster_min_points, cluster_max_points,
        dim, n_support_min, n_support_max, n_query, estimate, n_series,
        n_hutch, network_type, noise_std, augment_type, augment_moment, mean_range,
        mixture_type, slope_coeff, num_layers, prob_cat
):
    datasets = list(islice(
        dataset_generator(
            num_gaussians, cluster_min_points, cluster_max_points, dim,
            n_support_min, n_support_max, n_query, estimate, n_series,
            n_hutch, network_type, noise_std, augment_type, augment_moment,
            mean_range, mixture_type, slope_coeff, num_layers, prob_cat
        ),
        num_datasets)
    )
    return datasets
