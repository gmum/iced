from typing import Tuple

import torch
from nflows.flows import SimpleRealNVP as _SimpleRealNVP
from torch import Tensor
from torch.nn import functional as F

from src.flows.base import BaseGenModel
from src.prior.gaussian_mixture import GaussianMixture
from src.prior.scaler import MinMaxScalerTorch


class SimpleRealNVP(BaseGenModel):
    def __init__(
        self,
        gmm_data: GaussianMixture,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        use_volume_preserving=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):
        super(SimpleRealNVP, self).__init__()
        self.model = _SimpleRealNVP(
            features=features,
            hidden_features=hidden_features,
            num_layers=num_layers,
            num_blocks_per_layer=num_blocks_per_layer,
            use_volume_preserving=use_volume_preserving,
            activation=activation,
            dropout_probability=dropout_probability,
            batch_norm_within_layers=batch_norm_within_layers,
            batch_norm_between_layers=batch_norm_between_layers,
        )
        self.gmm = gmm_data

    def forward(self, x, context=None):
        points, log_det =  self.model._transform.forward(x)
        return points, log_det

    def fit(
            self,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            num_epochs: int = 100,
            learning_rate: float = 1e-3,
            patience: int = 20,
            eps: float = 1e-3,
            checkpoint_path: str = "best_model.pth",
    ):
        pass

    def predict_log_prob(self, dataloader) -> torch.Tensor:
        """
        Predict log probabilities for the given dataset using the context included in the dataset.
        """
        self.eval()
        log_probs = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                labels = labels.type(torch.float32)
                outputs = self(inputs, labels)
                log_probs.append(outputs)
        results = torch.concat(log_probs)

        assert len(dataloader.dataset) == len(results)
        return results

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def transform(self, X_support: Tensor, X_query: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        prior_log_support = self.gmm.log_prob(X_support, rescale=True)
        prior_log_query = self.gmm.log_prob(X_query, rescale=True)

        X_support, log_det_support = self(X_support)
        X_query, log_det_query = self(X_query)

        min_scaler = MinMaxScalerTorch(feature_range=(-1, 1))
        X_support = min_scaler.fit_transform(X_support)
        X_query = min_scaler.transform(X_query)
        self.scaler = min_scaler

        log_probs_support = prior_log_support - log_det_support
        log_probs_query = prior_log_query - log_det_query

        return X_support, X_query, log_probs_support, log_probs_query
