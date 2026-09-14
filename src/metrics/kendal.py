import torch
from scipy.stats import kendalltau

from src.metrics.abstract import Metric


class KendallTau(Metric):
    def __init__(self):
        super().__init__()

        self.name = "kendaltau"

    def compute(self, output: torch.Tensor, log_probs: torch.Tensor):
        tau, _ = kendalltau(output, log_probs)
        self.values.append(tau)
        return tau
