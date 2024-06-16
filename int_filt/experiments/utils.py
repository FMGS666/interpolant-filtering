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
    SimNLGTanh,
    SimNLGExp,
    IdentityPreproc,
    StandardizeSim,
)

from ..utils import ConfigData

NON_LINEARITIES = {
    "cos": SimNLGCos,
    "sin": SimNLGSin,
    "tan": SimNLGTan,
    "tanh": SimNLGTanh,
    "exp": SimNLGExp,
}

PREPROCESSING = {
    "none": IdentityPreproc,
    "sim": StandardizeSim,
}

def create_experiment(config: ConfigData) -> Experiment:
    if config["experiment"] == "ou":
        ## parsing configuration dictionary
        ## model
        sigma_x = config["sigma_x"]
        sigma_y = config["sigma_y"]
        beta = config["beta"]
        num_dims = config["num_dims"]
        num_sims = config["num_sims"]
        ## interpolant
        interpolant_method = config["interpolant_method"]
        epsilon = config["epsilon"]
        ## b-net 
        b_net_hidden_dims = config["b_net_hidden_dims"]
        b_net_activation = config["b_net_activation"]
        b_net_activate_final = config["b_net_activate_final"]
        b_net_amortized = config["b_net_amortized"]
        ## mc
        mc_config = config["mc_config"]
        ## logging 
        log = config["log"]
        logging_step = config["logging_step"]
        log_dir = config["log_dir"]
        writer = SummaryWriter(log_dir=os.path.join(log_dir, 'summary'))
        ## device
        device = config["device"]
        ## preprocessing
        preprocessing = config["preprocessing"]
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
        interpolant_config = {"method": interpolant_method, "epsilon": epsilon}
        interpolant = create_interpolant(interpolant_config)
        ## initializing models
        models_config = {
            "backbone": "mlp",
            "spatial_dims": num_dims,
            "b_net_hidden_dims": b_net_hidden_dims,
            "b_net_activation": b_net_activation,
            "b_net_activate_final": b_net_activate_final,
            "b_net_amortized": b_net_amortized,
            "device": device
        }
        models = create_models(models_config)
        b_net = models["b_net"]
        ## initializing preprocessing
        preprocessing_config = {
            "ssm": ssm
        }
        preprocessing = PREPROCESSING[preprocessing](preprocessing_config)
        ## initializing experiment
        experiment_config = {
            "interpolant": interpolant,
            "b_net": b_net, 
            "ssm": ssm,
            "log": log, 
            "logging_step": logging_step,
            "writer": writer,
            "mc_config": mc_config,
            "device": device,
            "preprocessing": preprocessing
        }
        experiment = OUExperiment(experiment_config)
    if config["experiment"] == "nlg":
        ## parsing configuration dictionary
        ## model
        sigma_x = config["sigma_x"]
        sigma_y = config["sigma_y"]
        beta = config["beta"]
        num_dims = config["num_dims"]
        num_sims = config["num_sims"]
        num_iters = config["num_iters"]
        step_size = config["step_size"]
        non_linearity = config["non_linearity"]
        ## interpolant
        interpolant_method = config["interpolant_method"]
        epsilon = config["epsilon"]
        ## b-net 
        b_net_hidden_dims = config["b_net_hidden_dims"]
        b_net_activation = config["b_net_activation"]
        b_net_activate_final = config["b_net_activate_final"]
        b_net_amortized = config["b_net_amortized"]
        ## mc
        mc_config = config["mc_config"]
        ## logging 
        log = config["log"]
        logging_step = config["logging_step"]
        log_dir = config["log_dir"]
        writer = SummaryWriter(log_dir=os.path.join(log_dir, 'summary'))
        ## device
        device = config["device"]
        ## preprocessing
        preprocessing = config["preprocessing"]
        ## initializing gaussian model 
        ssm_config = {
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "beta": beta,
            "num_dims": num_dims,
            "num_sims": num_sims,
            "num_iters": num_iters,
            "step_size": step_size
        }
        ssm = NON_LINEARITIES[non_linearity](ssm_config)
        ## initializing interpolant
        interpolant_config = {"method": interpolant_method, "epsilon": epsilon}
        interpolant = create_interpolant(interpolant_config)
        ## initializing models
        models_config = {
            "backbone": "mlp",
            "spatial_dims": num_dims,
            "b_net_hidden_dims": b_net_hidden_dims,
            "b_net_activation": b_net_activation,
            "b_net_activate_final": b_net_activate_final,
            "b_net_amortized": b_net_amortized,
            "device": device
        }
        models = create_models(models_config)
        b_net = models["b_net"]
        ## initializing preprocessing
        preprocessing_config = {
            "ssm": ssm
        }
        preprocessing = PREPROCESSING[preprocessing](preprocessing_config)
        ## initializing experiment
        experiment_config = {
            "interpolant": interpolant,
            "b_net": b_net, 
            "ssm": ssm,
            "log": log, 
            "logging_step": logging_step,
            "writer": writer,
            "mc_config": mc_config,
            "device": device,
            "preprocessing": preprocessing
        }
        experiment = NLGExperiment(experiment_config)
    return experiment