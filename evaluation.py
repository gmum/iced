import os
from datetime import datetime
from pathlib import Path

import torch
import wandb

from src.initialization import initialize
from src.evaluation import evaluate_probs
from src.loss.main_loss import get_loss_fn
from src.utils import evaluate_anomalies, evaluate_density, evaluate_augmentations

config, model = initialize()

model_name = Path(config.model_path).stem
wandb.init(project='ICED_eval', tags=['our', ], dir=config.output_dir, name=model_name)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plot_dir = os.path.join(config.output_dir, "plots", timestamp)
os.makedirs(plot_dir, exist_ok=True)

model = model.eval()

if config.testsets_path != "":
    filename = Path(config.testsets_path).stem
    wandb.run.tags += (filename,)

    test_datasets = torch.load(config.testsets_path)
    loss_fn = get_loss_fn(config.base_loss_type, config.sort_loss_type)
    evaluate_probs(model, test_datasets, config.device, loss_fn, "test")
else:
    evaluate_density(model, config.device, plot_dir, evaluate=True)


evaluate_anomalies(model, config.device, config.anomaly_stratify)
evaluate_augmentations(model, config.device)

wandb.finish()

