from typing import Callable

import torch
from torch import Tensor

from src.configs import BaseLossType, SortLossType
from src.loss.base import BaseLossFn
from src.loss.logistic import LogisticLossFn


def get_loss_fn(base_loss_type: BaseLossType, sort_loss_type: SortLossType) -> Callable[[Tensor, Tensor], Tensor]:
    match base_loss_type:
        case BaseLossType.MSE:
            base_loss_fn = BaseLossFn()
        case BaseLossType.NONE:
            base_loss_fn = lambda x, y: torch.tensor(0.0, device=x.device)

    match sort_loss_type:
        case SortLossType.NONE:
            sort_loss_fn = lambda x, y: torch.tensor(0, device=x.device)
        case SortLossType.LOGISTIC:
            sort_loss_fn = LogisticLossFn()

    def combined_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return base_loss_fn(output, target) + sort_loss_fn(output, target)

    return combined_loss
