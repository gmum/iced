import random

import torch

from src.configs import AugmentationType


def gaussian_noise_augmentation(X_query):
    cur_dim = X_query.shape[-1]
    noise_std = 7.76 * cur_dim ** (-1.955)

    noise = torch.randn_like(X_query) * noise_std
    return noise

def mixup_augmentation(X_support, X_query):
    lam = torch.empty(1).normal_(0.5, 0.1).item()
    idx = torch.randint(0, X_support.size(0), (X_query.size(0),))
    X_mix = lam * X_support[idx] + (1 - lam) * X_query

    return X_mix - X_query

def cutmix_augmentation(X_support, X_query):
    Bq, F = X_query.shape
    Bs = X_support.shape[0]

    ratio = torch.empty(1).uniform_(0.1, 0.5).item()
    k = int(F * ratio)
    cols = torch.randperm(F)[:k]
    idx = torch.randint(0, Bs, (Bq,))

    X_mix = X_query.clone()
    X_mix[:, cols] = X_support[idx][:, cols]

    return X_mix - X_query

def augment_data(X_support, X_query, augment_type):
    if augment_type == AugmentationType.RANDOM:
        cur_augment_type = random.choice([
            AugmentationType.GAUSSIAN_NOISE,
            AugmentationType.MIXUP, AugmentationType.CUTMIX]) # AugmentationType.UNIFORM_NOISE
    else:
        cur_augment_type = augment_type

    mask = torch.rand_like(X_query) < 0.1
    if cur_augment_type == AugmentationType.GAUSSIAN_NOISE:
        noise = gaussian_noise_augmentation(X_query)
    elif cur_augment_type == AugmentationType.MIXUP:
        noise = mixup_augmentation(X_support, X_query)
    else: # augment_type == AugmentationType.CUTMIX:
        noise = cutmix_augmentation(X_support, X_query)

    X_query = X_query + mask * noise
    return X_query
