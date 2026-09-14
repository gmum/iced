from abc import ABC, abstractmethod

import torch


class Metric(ABC):
    """
    Abstract base class for computing metrics.
    """

    def __init__(self):
        self.name = "abstract"
        self.values = []

    @abstractmethod
    def compute(self, output: torch.Tensor, log_probs: torch.Tensor):
        """
        Compute Kendall's Tau for a single batch.

        Args:
            output (list or array): Model outputs / scores
            log_probs (list or array): Corresponding log probabilities

        Returns:
            float: Metric value
        """
        pass

    def get_mean_value(self):
        """
        Return the mean over all computed batches.

        Returns:
            float: Mean metric value.
        """
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def reset(self):
        """
        Reset internal stored values.
        """
        self.values = []
