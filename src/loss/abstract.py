from abc import abstractmethod, ABC

import torch


class AbstractLossFn(ABC):
    @abstractmethod
    def __call__(self, output: torch.Tensor, target: torch.Tensor ) -> torch.Tensor:
        pass
