import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from omegaconf import OmegaConf

from src.loss.main_loss import get_loss_fn
from src.prior.datasets import generate_gmm_datasets_with_projected_points, GMMDataset
from src.initialization import initialize
from src.evaluation.density import evaluate_probs
from src.utils import evaluate_everything
from src.model.model_utils import get_cosine_schedule_with_warmup
from collections import defaultdict
import torch.multiprocessing as mp

if __name__ == '__main__':
    config, model = initialize()
    mp.set_start_method("spawn", force=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    checkpoint_dir = os.path.join(config.output_dir, "models", timestamp)
    plot_dir = os.path.join(config.output_dir, "plots", timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    val_datasets = generate_gmm_datasets_with_projected_points(
        config.num_test_datasets, config.num_gaussians, config.cluster_min_points, config.cluster_max_points,
        config.dim, config.n_support_min, config.n_support_max, config.n_query, config.estimate, config.n_series,
        config.n_hutch, config.network_type, config.noise_std, config.augment_type, config.augment_moment,
        config.mean_range, config.mixture_type, config.slope_coeff, config.num_layers, config.prob_cat
    )

    test_datasets = generate_gmm_datasets_with_projected_points(
        config.num_test_datasets, config.num_gaussians, config.cluster_min_points, config.cluster_max_points,
        config.dim, config.n_support_min, config.n_support_max, config.n_query, config.estimate, config.n_series,
        config.n_hutch, config.network_type, config.noise_std, config.augment_type, config.augment_moment,
        config.mean_range, config.mixture_type, config.slope_coeff, config.num_layers, config.prob_cat
    )

    wandb.init(project='ICED', dir=config.output_dir)
    wandb.config.update(OmegaConf.to_container(config))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 0,
        config.nr_epochs*config.num_train_datasets // config.accum_steps,
    )
    if config.model_path != "":
        checkpoint = torch.load(config.model_path, map_location=torch.device('cpu'))
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            print(e)

    train_dataset = GMMDataset(
        config.num_train_datasets, config.num_gaussians, config.cluster_min_points, config.cluster_max_points,
        config.dim, config.n_support_min, config.n_support_max, config.n_query, config.estimate, config.n_series,
        config.n_hutch, config.network_type, config.noise_std, config.augment_type, config.augment_moment,
        config.mean_range, config.mixture_type, config.slope_coeff, config.num_layers, config.prob_cat
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, pin_memory=True,
                              num_workers=8, persistent_workers=True)
    loss_fn = get_loss_fn(config.base_loss_type, config.sort_loss_type)

    for i in tqdm(range(config.nr_epochs)):
        model.train()

        epoch_loss = 0
        train_metrics = defaultdict(lambda: defaultdict(int))
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (X_support, X_query, log_probs) in enumerate(train_loader):
            X_support, X_query, log_probs = X_support.to(config.device), X_query.to(config.device), log_probs.to(config.device)
            X_support = X_support.transpose(0, 1)
            X_query = X_query.transpose(0, 1)

            output = model(X_support, X_query)
            loss = loss_fn(output, log_probs)
            epoch_loss += loss.item()

            loss = loss / config.accum_steps
            loss.backward()

            if (batch_idx + 1) % config.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        epoch_loss /= config.num_train_datasets

        wandb.log({"train/loss_mean": epoch_loss})

        if i % 100 == 0:
            evaluate_probs(model, val_datasets, config.device, loss_fn, "val")

    evaluate_probs(model, test_datasets, config.device, loss_fn, "test")
    evaluate_everything(model, config, plot_dir)

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)

    wandb.finish()
