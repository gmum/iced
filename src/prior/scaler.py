import torch


class MinMaxScalerTorch:
    def __init__(self, feature_range=(0.0, 1.0), dim=0, eps=1e-8):
        self.min, self.max = feature_range
        self.dim = dim
        self.eps = eps
        self.data_min = None
        self.data_max = None

    def fit(self, x: torch.Tensor):
        self.data_min = x.min(dim=self.dim, keepdim=True).values
        self.data_max = x.max(dim=self.dim, keepdim=True).values
        return self

    def transform(self, x: torch.Tensor):
        x_std = (x - self.data_min) / (self.data_max - self.data_min + self.eps)
        return x_std * (self.max - self.min) + self.min

    def inverse_transform(self, x: torch.Tensor):
        x_std = (x - self.min) / (self.max - self.min)
        return x_std * (self.data_max - self.data_min + self.eps) + self.data_min

    def fit_transform(self, x: torch.Tensor):
        return self.fit(x).transform(x)