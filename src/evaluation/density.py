import torch
import wandb

from src.metrics.kendal import KendallTau
from src.metrics.loss import LossMetric
from src.metrics.pairwise_order import PairwiseAccuracyMetric
from src.model.iced import ICED


def evaluate_probs(model: ICED, datasets, device, loss_fn, prefix="val"):
    model.eval()
    metrics = [LossMetric(loss_fn), KendallTau(), PairwiseAccuracyMetric()]

    for batch_idx, (X_support, X_query, log_probs) in enumerate(datasets):
        X_support, X_query = X_support.to(device), X_query.to(device)
        X_support = X_support.unsqueeze(1)
        X_query = X_query.unsqueeze(1)

        with torch.no_grad():
            output = model(X_support, X_query)
        output = output.squeeze(0).detach().cpu()

        batch_metrics = {}
        for metric in metrics:
            metric_value = metric.compute(output, log_probs)
            batch_metrics[f"{prefix}/{metric.name}_set_{batch_idx}"] = metric_value
        wandb.log(batch_metrics)

    metrics_log_dict = {
        f"{prefix}/{metric.name}": metric.get_mean_value() for metric in metrics
    }
    wandb.log(metrics_log_dict)
