import torch

from src.metrics.abstract import Metric


class LossMetric(Metric):
    """
    Metric wrapper for tracking a loss function over batches.

    Args:
        loss_fn: Callable with signature loss_fn(output, log_probs)
                 returning a tensor with `.item()`
    """

    def __init__(self, loss_fn):
        super().__init__()

        self.loss_fn = loss_fn
        self.name = "loss"

    def compute(self, output: torch.Tensor, log_probs: torch.Tensor):
        """
        Compute loss for a single batch.
        """
        loss = self.loss_fn(output, log_probs).item()
        self.values.append(loss)
        return loss
