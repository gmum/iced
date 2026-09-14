import torch

from src.loss.abstract import AbstractLossFn


class BaseLossFn(AbstractLossFn):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inside_mask = (target >= -10) & (target <= 10)
        outside_mask = ~inside_mask

        loss = torch.tensor(0., device=output.device)

        if inside_mask.any():
            loss += self.mse(output[inside_mask], target[inside_mask])

        if outside_mask.any():
            lower_violation = (target < -10) & (output > -10)
            upper_violation = (target > 10) & (output < 10)

            penalty = torch.zeros_like(output)
            penalty[lower_violation] = (output[lower_violation] - (-11)) ** 2
            penalty[upper_violation] = (output[upper_violation] - 11) ** 2

            loss += penalty.mean()

        return loss
