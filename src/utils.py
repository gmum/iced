import random
import numpy as np
import torch
import wandb

from src.configs import BaseLossType, SortLossType, GMMConfig
from src.evaluation import evaluate_probs, run_evaluate
from src.evaluation.augmentation import evaluate_augmentation
from src.loss.main_loss import get_loss_fn
from src.model.iced import ICED

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate_everything(model: ICED, config: GMMConfig, plot_dir: str):
    model = model.eval()
    evaluate_anomalies(model, config.device, config.anomaly_stratify)
    evaluate_density(model, config.device, plot_dir, config.base_loss_type, config.sort_loss_type)
    #evaluate_augmentations(model, config.device)

def evaluate_anomalies(model: ICED, device:str, anomaly_stratify:bool):
    print("Evalauate anomaly score")
    for anomaly_stratify in [True, False]:
        run_evaluate(model, anomaly_stratify=anomaly_stratify, max_dim=model.input_dim, seedlist=[0, 1, 2, 3, 4])

def evaluate_augmentations(model: ICED, device:str):
    for dataset_name in ["protein", "fourier", "biodeg", "steel", "stock", "energy", "collins", "texture"]:
        evaluate_augmentation(model, dataset_name, device)

def evaluate_density(
        model, device, plot_dir, base_loss_type: BaseLossType = BaseLossType.MSE,
        sort_loss_type: SortLossType = SortLossType.LOGISTIC, evaluate = False):
    for dataset_name in density_sets:
        path = f"density_sets/{dataset_name}.pt"
        test_datasets = torch.load(path)
        loss_fn = get_loss_fn(base_loss_type, sort_loss_type)
        if evaluate:
            run_name = wandb.run.name
            project = wandb.run.project
            wandb.finish()
            wandb.init(
                project=project,
                name=f"{run_name}",
                tags=["our", f"{dataset_name}"],
                reinit=True
            )
            evaluate_probs(model, test_datasets, device, loss_fn, "test")
        else:
            evaluate_probs(model, test_datasets, device, loss_fn, f"test_{dataset_name}")

density_sets = [
    "net_NONE_aug_RANDOM_moment_NONE_mix_GAUSSIAN_cat_0.3",
    "net_NONE_aug_RANDOM_moment_NONE_mix_HEAVY_TAIL_cat_0.3",
    "net_NONE_aug_RANDOM_moment_PRE_TRANSFORM_mix_RANDOM_cat_0.3",
    "net_RANDOM_aug_RANDOM_moment_NONE_mix_RANDOM_cat_0.0",
    "net_RANDOM_aug_RANDOM_moment_NONE_mix_RANDOM_cat_0.3",
    "net_RANDOM_aug_RANDOM_moment_PRE_TRANSFORM_mix_RANDOM_cat_0.0",
    "net_RANDOM_aug_RANDOM_moment_PRE_TRANSFORM_mix_RANDOM_cat_0.3",
]
real_world_datasets = [
    6, 11, 15, 18, 22, 32, 37, 53, 2074, 3902, 3903, 3904, 3913, 3917, 3918, 9946, 9952, 9957, 9960,
    10093, 10101, 14969, 125922, 146817, 146819, 146820, 146822, 167119, 167120
]
