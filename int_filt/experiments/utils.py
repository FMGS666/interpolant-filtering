"""
File containing utility functions for running experiments
"""
import torch
import os

from tensorboardX import SummaryWriter

from .common import Experiment
from .ornstein_uhlenbeck import OUExperiment
from .non_linear_gaussian import NLGExperiment

from ..src import (
    create_interpolant, 
    create_models, 
    SimOrnsteinUhlenbeck, 
    SimNLGCos, 
    SimNLGSin, 
    SimNLGTan,
    SimNLGExp,
    IdentityPreproc,
    StandardizeSim,
)

from ..utils import ConfigData

NON_LINEARITIES = {
    "cos": SimNLGCos,
    "sin": SimNLGSin,
    "tan": SimNLGTan,
    "exp": SimNLGExp,
}

PREPROCESSING = {
    "none": IdentityPreproc,
    "sim": StandardizeSim,
}

def create_experiment(config: ConfigData) -> Experiment:
    if config["experiment"] == "ou":
        ## logging 
        writer = None
        if config["log_results"]:
            writer = SummaryWriter(log_dir=os.path.join(config["log_dir"], 'summary'))
        ## initializing ou-model 
        ssm_config = {
            "sigma_x": config["sigma_x"],
            "sigma_y": config["sigma_y"],
            "beta": config["beta"],
            "num_dims": config["num_dims"],
            "num_sims": config["num_sims"]
        }
        ssm = SimOrnsteinUhlenbeck(ssm_config)
        ## initializing interpolant
        interpolant_config = {"method": config["interpolant_method"], "epsilon": config["epsilon"]}
        interpolant = create_interpolant(interpolant_config)
        ## initializing models
        models_config = {
            "backbone": config["backbone"],
            "spatial_dims": config["num_dims"],
            "b_net_hidden_dims": config["b_net_hidden_dims"],
            "b_net_activation": config["b_net_activation"],
            "b_net_activate_final": config["b_net_activate_final"],
            "b_net_amortized": config["b_net_amortized"],
            "device": config["device"]
        }
        models = create_models(models_config)
        b_net = models["b_net"]
        ## initializing preprocessing
        preprocessing_config = {
            "ssm": ssm
        }
        preprocessing = PREPROCESSING[config["preprocessing"]](preprocessing_config)
        ## initializing experiment
        experiment_config = {
            "interpolant": interpolant,
            "b_net": b_net, 
            "ssm": ssm,
            "preprocessing": preprocessing,
            "writer": writer,
            "log_results": config["log_results"], 
            "logging_step": config["logging_step"],
            "mc_config": config["mc_config"],
            "device": config["device"],
        }
        experiment = OUExperiment(experiment_config)
    if config["experiment"] == "nlg":
        ## logging 
        writer = None
        if config["log_results"]:
            writer = SummaryWriter(log_dir=os.path.join(config["log_dir"], 'summary'))
        ## device
        device = config["device"]
        ## preprocessing
        preprocessing = config["preprocessing"]
        ## initializing gaussian model 
        ssm_config = {
            "sigma_x": config["sigma_x"],
            "sigma_y": config["sigma_y"],
            "beta": config["beta"],
            "num_dims": config["num_dims"],
            "num_sims": config["num_sims"],
            "num_iters": config["num_iters"],
            "step_size": config["step_size"]
        }
        ssm = NON_LINEARITIES[config["non_linearity"]](ssm_config)
        ## initializing interpolant
        interpolant_config = {"method": config["interpolant_method"], "epsilon": config["epsilon"]}
        interpolant = create_interpolant(interpolant_config)
        ## initializing models
        models_config = {
            "backbone": config["backbone"],
            "spatial_dims": config["num_dims"],
            "b_net_hidden_dims": config["b_net_hidden_dims"],
            "b_net_activation": config["b_net_activation"],
            "b_net_activate_final": config["b_net_activate_final"],
            "b_net_amortized": config["b_net_amortized"],
            "device": config["device"]
        }
        models = create_models(models_config)
        b_net = models["b_net"]
        ## initializing preprocessing
        preprocessing_config = {
            "ssm": ssm
        }
        preprocessing = PREPROCESSING[config["preprocessing"]](preprocessing_config)
        ## initializing experiment
        experiment_config = {
            "interpolant": interpolant,
            "b_net": b_net, 
            "ssm": ssm,
            "preprocessing": preprocessing,
            "writer": writer,
            "log_results": config["log_results"], 
            "logging_step": config["logging_step"],
            "mc_config": config["mc_config"],
            "device": config["device"],
        }
        experiment = NLGExperiment(experiment_config)
    return experiment