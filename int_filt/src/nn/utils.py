"""
File containing the implementation of utility functions 
for proper allocation of stochastic interpolants
"""
import torch

from .backbones import MLP
from .models import B_Net, C_Net

from ...utils import ConfigData, ModelData

def create_models(config: ConfigData) -> ModelData:
    if config["backbone"] == "mlp":
        ## defining inputs dims
        input_dims = config["spatial_dims"]*3 + 1 if config["b_net_amortized"] else config["spatial_dims"]*2 + 1
        ## initializing $b$ backbone
        b_backbone_config = {
            "input_dim": input_dims,
            "hidden_dims": config["b_net_hidden_dims"],
            "output_dim": config["spatial_dims"],
            "activation": config["b_net_activation"],
            "activate_final": config["b_net_activate_final"],
        }
        b_backbone = MLP(b_backbone_config)
        ## initializing $b$ model
        b_net_config = {"backbone": b_backbone, "amortized": config["b_net_amortized"]}
        b_net = B_Net(b_net_config)
        b_net.to(config["device"])
    models = {"b_net": b_net}
    return models

def create_controlled_models(config: ConfigData) -> ModelData:
    if config["backbone"] == "mlp":
        ## defining inputs dims
        b_input_dims = config["spatial_dims"]*3 + 1 if config["b_net_amortized"] else config["spatial_dims"]*2 + 1
        c_input_dims = config["spatial_dims"]*3 + 1 
        ## initializing $b$ backbone
        b_backbone_config = {
            "input_dim": b_input_dims,
            "hidden_dims": config["b_net_hidden_dims"],
            "output_dim": config["spatial_dims"],
            "activation": config["b_net_activation"],
            "activate_final": config["b_net_activate_final"]
        }
        b_backbone = MLP(b_backbone_config)
        ## initializing $c$ backbone
        c_backbone_config = {
            "input_dim": c_input_dims,
            "hidden_dims": config["c_net_hidden_dims"],
            "output_dim": config["spatial_dims"],
            "activation": config["c_net_activation"],
            "activate_final": config["c_net_activate_final"]
        }
        c_backbone = MLP(c_backbone_config)
        ## initializing $b$ and $c$ models
        b_net_config = {"backbone": b_backbone, "amortized": False}
        c_net_config = {"backbone": c_backbone}
        b_net = B_Net(b_net_config)
        c_net = C_Net(c_net_config)
        b_net.to(config["device"])
        c_net.to(config["device"])
    models = {"b_net": b_net, "c_net": c_net}
    return models