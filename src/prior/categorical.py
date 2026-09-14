import random

import torch

def build_safe_class_counts(d_num, d_cat, max_dim=50, min_c=2, max_c=5):
    """
    Ensures:
        d_num + sum(class_counts) <= max_dim
    """

    max_cat_dim = max_dim - d_num
    class_counts = []
    remaining = max_cat_dim
    remaining = 1000

    for j in range(d_cat):
        remaining_features = d_cat - j

        # guarantee feasibility for remaining features
        max_allowed = min(max_c, remaining - (remaining_features - 1) * min_c)
        min_allowed = min_c

        if max_allowed < min_allowed:
            continue

        Cj = random.randint(min_allowed, max_allowed)
        class_counts.append(Cj)
        remaining -= Cj

    return class_counts

def sample_dirichlet_categorical(
    class_counts,
    n_support,
    n_query,
    alpha=1.0,
    device="cpu"
):
    """
    Independent categorical features:
        p_j ~ Dirichlet(alpha)
        x_j ~ Categorical(p_j)
    """

    # sample per-feature categorical distributions
    p = []

    for j, Cj in enumerate(class_counts):
        p_j = torch.distributions.Dirichlet(
            alpha * torch.ones(Cj, device=device)
        ).sample()

        p.append(p_j)

    def sample(n):
        cols = []

        for probs in p:
            cols.append(
                torch.distributions.Categorical(probs).sample((n,))
            )

        return torch.cat([c.unsqueeze(1) for c in cols], dim=1)

    X_support_cat = sample(n_support)
    X_query_cat = sample(n_query)

    def log_prob(X_cat):
        log_probs = torch.zeros(X_cat.shape[0], device=device)

        for j, probs in enumerate(p):
            log_probs += probs.log()[X_cat[:, j]]

        return log_probs

    log_probs_support = log_prob(X_support_cat)
    log_probs_query = log_prob(X_query_cat)

    return X_support_cat, X_query_cat, log_probs_support, log_probs_query
