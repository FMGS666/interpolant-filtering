import torch
import os

from pathlib import Path

from .experiments import create_experiment

from .utils.config import configuration
from .utils.utils import ensure_reproducibility
from .utils.distributions import GaussianObservationModel

ACTIVATIONS = {
    "relu": torch.nn.ReLU()
}

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adam-w": torch.optim.AdamW
}

SCHEDULERS = {
    "none": None
}

OBSERVATION_MODELS = {
    "gaussian": GaussianObservationModel
}

def configure_observation_model(observation_model_class, args):
    if observation_model_class == "gaussian":
        observation_model_config = {"sigma_y": args["sigma_y"]}
    return observation_model_config

if __name__ == "__main__":
    ## parsing arguments
    args = configuration()
    ## retrieving activations
    args.b_net_activation = ACTIVATIONS[args.b_net_activation]
    ## creating experiment
    args = vars(args)
    ## adding mc configuration
    args["mc_config"] = {"num_samples": args["num_samples"]}
    ## prepare for training drift
    b_net_num_grad_step = args["b_net_num_grad_steps"]
    b_net_optimizer = args["b_net_optimizer"]
    b_net_scheduler = args["b_net_scheduler"]
    b_net_lr = args["b_net_lr"]

    ## dump dir 
    dump_dir = args["dump_dir"]
    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)

    ## reproducibility
    random_seed = args["random_seed"]
    ensure_reproducibility(random_seed)
    
    ## initilizing observation model
    observation_model_class = args["observation_model"]
    observation_model_config = configure_observation_model(observation_model_class, args)
    observation_model = OBSERVATION_MODELS[observation_model_class](observation_model_config)

    ## creating experiment
    experiment = create_experiment(args)

    ## initializing optimizer
    b_net_optimizer = OPTIMIZERS[b_net_optimizer](experiment.b_net.backbone.parameters(), lr = b_net_lr)

    ## constructing optimization config dictionary
    b_net_optim_config = {
        "num_grad_steps": b_net_num_grad_step,
        "optimizer": b_net_optimizer,
        "scheduler": b_net_scheduler
    }

    ## training b_net 
    experiment.train(b_net_optim_config)
    ## saving the weights
    torch.save(experiment.b_net.state_dict(), os.path.join(dump_dir, "b_net.pt"))