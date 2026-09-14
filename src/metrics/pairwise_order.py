import torch

from src.metrics.abstract import Metric


class PairwiseAccuracyMetric(Metric):
    """
    Pairwise AUROC-style accuracy.
    """

    def __init__(self):
        super().__init__()

        self.name = "pairwise_accuracy"

    def compute(self, output: torch.Tensor, log_probs: torch.Tensor):
        """
        Args:
            output: ground-truth scores (Tensor, shape [n])
            log_probs: predicted scores (Tensor, shape [n])
        """

        y = output.detach()
        s = log_probs.detach()

        n = y.numel()
        if n < 2:
            self.values.append(0.0)
            return 0.0

        # Pairwise differences (broadcasting)
        dy = y.unsqueeze(0) - y.unsqueeze(1)
        ds = s.unsqueeze(0) - s.unsqueeze(1)

        # Only consider upper triangular (i < j)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=y.device), diagonal=1)

        dy = dy[mask]
        ds = ds[mask]

        product = dy * ds

        ties = product == 0
        concordant = product > 0
        discordant = product < 0

        C = concordant.sum().float()
        T = ties.sum().float()

        total_pairs = n * (n - 1) / 2

        pa = (C + 0.5 * T) / total_pairs
        pa = pa.item()

        self.values.append(pa)
        return pa
