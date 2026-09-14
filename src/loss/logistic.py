import torch

from src.loss.abstract import AbstractLossFn

class LogisticLossFn(AbstractLossFn):
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output = output.view(-1)
        target = target.view(-1)

        pred_diff = output.unsqueeze(1) - output.unsqueeze(0)
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)

        mask = target_diff > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=output.device)

        pred_diff = pred_diff[mask]
        loss = torch.nn.functional.softplus(-pred_diff)

        return loss.mean()
