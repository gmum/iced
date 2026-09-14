from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerEncoder

from src.model.icl_encoder import TabICLEncoder
from src.model.layer import TransformerEncoderLayer
from src.model.model_utils import SeqBN, bool_mask_to_att_mask
from sklearn.preprocessing import MinMaxScaler


class ICED(nn.Module):
    def __init__(self, encoder, ninp, nhead, nhid, nlayers,
                 dropout=0.0, *, decoder=None, input_normalization=False, pre_norm=False,
                 activation='gelu', recompute_attn=False, full_attention=False,
                 all_layers_same_init=False, efficient_eval_masking=True, input_dim=50):
        super().__init__()
        self.model_type = 'Transformer'
        self.icl_encoder = TabICLEncoder()

        encoder_layer_creator = lambda: TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation=activation,
                                                                pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.transformer_encoder = TransformerEncoder(encoder_layer_creator(), nlayers)\
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.ninp = ninp
        self.encoder = encoder

        n_out = 1
        self.decoder = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.GELU(),
            nn.Linear(nhid, n_out)
        )
        #self.out_layer = nn.Linear(nhid, n_out, bias=False)

        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.efficient_eval_masking = efficient_eval_masking
        self.full_attention = full_attention

        self.n_out = n_out
        self.nhid = nhid
        self.input_dim = input_dim

        #self.cluster_centers = nn.Parameter(torch.randn(n_clusters, 1, ninp))

        self.init_weights()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('efficient_eval_masking', False)

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        train_size = sz-query_size
        mask = torch.zeros(sz, sz) == 0
        mask[:, train_size:].zero_()
        mask |= torch.eye(sz) == 1
        return bool_mask_to_att_mask(mask)

    def init_weights(self):
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
            for attn in attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)

    def forward(self, x_sup, x_query, augmentations=False):
        x_sup, x_query = self.icl_encoder(x_sup, x_query, augmentations=augmentations)

        x_sup = self.encoder(x_sup)
        x_query = self.encoder(x_query)
        src_mask = None

        if src_mask is None:
            full_len = len(x_sup) + len(x_query) #+ len(self.cluster_centers)
            if self.full_attention:
                src_mask = bool_mask_to_att_mask(torch.ones((full_len, full_len), dtype=torch.bool)).to(x_sup.device)
            elif self.efficient_eval_masking:
                src_mask = len(x_sup)
            else:
                src_mask = self.generate_D_q_matrix(full_len, 0).to(x_sup.device)

        src = torch.cat([x_sup, x_query], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        output = self.transformer_encoder(src, src_mask)

        output = self.decoder(output[len(x_sup):])
#
        #if self.distance_based_logit:
        #    output = -torch.linalg.norm(output.unsqueeze(-1) - self.out_layer.weight.T, dim=-2)**2

        return output.transpose(0, 1).squeeze(-1)

    def fit(self, X):
        X = torch.from_numpy(X).float()
        X = X.unsqueeze(1)
        self.X = X

    def score_samples(self, X_test, device='cuda'):
        X_test = torch.from_numpy(X_test).float()
        X_test = X_test.unsqueeze(1)

        with torch.no_grad():
            score = self.forward(self.X.to(device), X_test.to(device)).squeeze(0).cpu()
        del self.X

        return score

    def generate(self, X_sup, y_sup, num_samples: int = 50, sgld_steps: int = 200,
                 starting_point_noise_std: float = 0.01, sgld_step_size: float = 0.1,
                 sgld_noise_std: float = 0.01, device: str = 'cuda'):

        data_dim = X_sup.shape[1]
        X_sup = torch.from_numpy(X_sup).float()

        # Copied from TabEBM
        unique_classes = np.unique(y_sup)
        synthetic_data_per_class = {}

        for target_class in unique_classes:
            # Pre-allocate noise tensor for all SGLD steps (memory optimization)
            class_size = int((np.sum(y_sup == target_class) / len(y_sup)) * num_samples)
            sample_shape = (class_size, 1, X_sup.shape[1])
            noise_shape = (sgld_steps, *sample_shape)

            X_ebm = X_sup[y_sup == target_class]
            y_ebm = np.zeros((X_ebm.shape[0], ))

            # Convert to tensors with proper device placement
            X_ebm = X_ebm.to(device)
            y_ebm = torch.from_numpy(y_ebm).long().to(device)

            real_sample_mask = y_ebm == 0
            real_samples = X_ebm[real_sample_mask]

            # Efficient random sampling
            num_real_samples = real_samples.shape[0]
            start_indices = torch.randint(0, num_real_samples, (class_size,), device=device)
            X_start = real_samples[start_indices]

            # Add noise to starting points (vectorized operation)
            if starting_point_noise_std > 0:
                noise = torch.randn_like(X_start, device=device) * starting_point_noise_std
                X_start = X_start + noise

            X_ebm = X_ebm.unsqueeze(1)
            X_sgld_tensor = X_start.to(device).unsqueeze(1).requires_grad_(True)
            noise_tensor = torch.randn(noise_shape, device=device, dtype=X_sgld_tensor.dtype)

            for t in range(sgld_steps):
                # Clear previous gradients
                if X_sgld_tensor.grad is not None:
                    X_sgld_tensor.grad.zero_()

                # Forward pass to compute energy
                energy = self.forward(X_ebm, X_sgld_tensor, augmentations=True).squeeze(0)

                total_energy = energy.sum() / X_sgld_tensor.shape[0]

                # Backward pass
                total_energy.backward()

                # SGLD update with pre-computed noise
                with torch.no_grad():
                    X_sgld_updated = (
                            X_sgld_tensor - sgld_step_size * X_sgld_tensor.grad + sgld_noise_std * noise_tensor[t]
                    )

                    # Update tensor in-place to maintain gradient tracking
                    X_sgld_tensor = X_sgld_updated.requires_grad_(True)

            X_sgld_tensor = X_sgld_tensor.squeeze(1)
            # Store results
            synthetic_data_per_class[f"class_{int(target_class)}"] =  X_sgld_tensor[:, :data_dim].detach().cpu().numpy()

        self.eval()
        return synthetic_data_per_class

class TransformerEncoderDiffInit(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer_creator: a function generating objects of TransformerEncoderLayer class without args (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
