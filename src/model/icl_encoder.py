import torch
from tabicl import InferenceConfig
from tabicl._model import TabICL
from tabicl._sklearn.preprocessing import TransformToNumerical

class TabICLEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        checkpoint = torch.load("checkpoints/tabicl/tabicl-classifier-v2-20260212.ckpt", map_location="cpu", weights_only=True)
        self.model = TabICL(**checkpoint["config"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def forward(self, X_support, X_query, augmentations=False):
        if augmentations:
            self.model.train()

        device = X_support.device
        X_support = X_support.cpu().permute(1, 0, 2)
        X_query = X_query.cpu().permute(1, 0, 2)

        inference_config = InferenceConfig()
        X = torch.cat([X_support, X_query], dim=1).to(device)
        y_support = torch.zeros(X_support.shape[0], X_support.shape[1]).to(device)
        train_size = X_support.shape[1]

        representations = self.model.row_interactor(
            self.model.col_embedder(
                X,
                y_support,
                feature_shuffles=None,
                mgr_config=inference_config.COL_CONFIG,
            ),
            mgr_config=inference_config.ROW_CONFIG,
        )

        representations = representations.permute(1, 0, 2)
        rep_support, rep_query = representations[:train_size], representations[train_size:]

        return rep_support, rep_query
