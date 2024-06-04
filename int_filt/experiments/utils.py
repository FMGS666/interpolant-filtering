"""
File containing utility functions for running experiments
"""
import torch
import os

from tensorboardX import SummaryWriter

from .common import Experiment
from .ornstein_uhlenbeck import OUExperiment

from ..src import create_interpolant, create_models, SimOrnsteinUhlenbeck
from ..utils import ConfigData

def create_experiment(config: ConfigData) -> Experiment:
    if config["experiment"] == "ornstein-uhlenbeck":
        ## parsing configuration dictionary
        ## model
        sigma_x = config["sigma_x"]
        sigma_y = config["sigma_y"]
        beta = config["beta"]
        num_dims = config["num_dims"]
        num_sims = config["num_sims"]
        ## interpolant
        interpolant_method = config["interpolant_method"]
        ## b-net 
        b_net_hidden_dims = config["b_net_hidden_dims"]
        b_net_activation = config["b_net_activation"]
        b_net_activate_final = config["b_net_activate_final"]
        b_net_amortized = config["b_net_amortized"]
        ## mc
        mc_config = config["mc_config"]
        ## logging 
        log_dir = config["log_dir"]
        writer = SummaryWriter(log_dir=os.path.join(log_dir, 'summary'))
        ## initializing ou-model 
        ssm_config = {
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "beta": beta,
            "num_dims": num_dims,
            "num_sims": num_sims
        }
        ssm = SimOrnsteinUhlenbeck(ssm_config)
        ## initializing interpolant
        interpolant_config = {"method": interpolant_method}
        interpolant = create_interpolant(interpolant_config)
        ## initializing models
        models_config = {
            "backbone": "mlp",
            "spatial_dims": num_dims,
            "b_net_hidden_dims": b_net_hidden_dims,
            "b_net_activation": b_net_activation,
            "b_net_activate_final": b_net_activate_final,
            "b_net_amortized": b_net_amortized
        }
        models = create_models(models_config)
        b_net = models["b_net"]
        ## initializing experiment
        experiment_config = {
            "interpolant": interpolant,
            "b_net": b_net, 
            "ssm": ssm,
            "writer": writer,
            "mc_config": mc_config
        }
        experiment = OUExperiment(experiment_config)
    return experiment