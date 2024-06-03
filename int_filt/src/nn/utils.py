"""
File containing the implementation of utility functions 
for proper allocation of stochastic interpolants
"""
import torch

from .backbones import MLP
from .models import B_Net

from ...utils import ConfigData, ModelData

def create_models(config: ConfigData) -> ModelData:
    if config["backbone"] == "mlp":
        ## parsing configuration dictionary
        spatial_dims = config["spatial_dims"]
        b_net_hidden_dims = config["b_net_hidden_dims"]
        b_net_activation = config["b_net_activation"]
        b_net_activate_final = config["b_net_activate_final"]
        ## initializing $b$ backbone
        b_backbone_config = {
            "input_dim": spatial_dims*3 + 1,
            "hidden_dims": b_net_hidden_dims,
            "output_dim": spatial_dims,
            "activation": b_net_activation,
            "activate_final": b_net_activate_final
        }
        b_backbone = MLP(b_backbone_config)
        ## initializing $b$ and $c$ models
        b_net_config = {"backbone": b_backbone}
        b_net = B_Net(b_net_config)
    models = {"b_net": b_net}
    return models