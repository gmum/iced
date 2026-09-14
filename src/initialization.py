from typing import cast, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from src.configs import GMMConfig
from src.model import encoders
from src.model.iced import ICED
from src.utils import setup_seed


def config_initialization() -> GMMConfig:
    base_config = OmegaConf.structured(GMMConfig)
    arg_config = OmegaConf.from_cli()

    config = cast(GMMConfig, OmegaConf.merge(base_config, arg_config))
    print(OmegaConf.to_yaml(config))

    return config


def model_initialization(config: GMMConfig) -> ICED:
    input_dim = 512
    encoder = encoders.Linear(input_dim, config.embed_dim, replace_nan_by_zero=True)

    model = ICED(
        encoder,config.embed_dim, config.n_head, config.hid_dim,
        config.n_layers, config.dropout, efficient_eval_masking=True,
        input_dim=config.dim,
    )
    model = model.to(config.device)
    return model


def initialize() -> Tuple[GMMConfig, ICED]:
    config = config_initialization()
    setup_seed(config.seed)

    model = model_initialization(config)
    if config.model_path != "":
        checkpoint = torch.load(config.model_path, map_location=torch.device('cpu'))
        ckpt_state = checkpoint["model"]
        model_state = model.state_dict()
        matched_state = {}

        for name, weight in ckpt_state.items():
            if name in model_state:
                if weight.shape == model_state[name].shape:
                    matched_state[name] = weight
                else:
                    print(
                        f"Skipping {name}: "
                        f"checkpoint {tuple(weight.shape)} != "
                        f"model {tuple(model_state[name].shape)}"
                    )

        model.load_state_dict(matched_state, strict=False)
    print(f"Params ", np.sum([p.numel() for p in model.parameters()]) / 1e6, "M")

    return config, model
