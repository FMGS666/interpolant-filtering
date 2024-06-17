import torch
import os

from pathlib import Path

from .experiments import create_experiment

from .utils import (
    configuration, 
    ensure_reproducibility, 
    dump_config
)

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

DEVICES = {
    "cpu": torch.device("cpu"),
    "cuda": torch.device("cuda")
}

if __name__ == "__main__":
    ## parsing arguments
    config = configuration()
    config = vars(config)
    ## displaying current arguments
    print(args)
    ## optional logging
    if config["log_results"]:
        ## creating dump dir 
        path = Path(config["dump_dir"])
        path.mkdir(parents=True, exist_ok=True)
        dump_config(config, os.path.join(config["dump_dir"], "config.json"))
    ## retrieving activation and device
    config["b_net_activation"] = ACTIVATIONS[config["b_net_activation"]]
    config["device"] = DEVICES[config["device"]]
    ## adding mc configuration
    config["mc_config"] = {"num_mc_samples": config["num_mc_samples"]}
    ## setting reproducibility
    ensure_reproducibility(config["random_seed"])
    ## creating experiment
    experiment = create_experiment(config)
    ## initializing optimizer and scheduler
    b_net_optimizer = OPTIMIZERS[config["b_net_optimizer"]](experiment.b_net.backbone.parameters(), lr = config["b_net_lr"])
    b_net_scheduler = SCHEDULERS[config["b_net_scheduler"]]
    if b_net_scheduler is not None:
        b_net_scheduler = b_net_scheduler(b_net_optimizer, config["b_net_num_grad_steps"])
    ## constructing optimization config dictionary
    b_net_optim_config = {
        "optimizer": b_net_optimizer,
        "scheduler": b_net_scheduler,
        "num_grad_steps": config["b_net_num_grad_steps"],
    }
    ## training b_net 
    experiment.train(b_net_optim_config)
    ## optional logging
    if config["log_results"]:
        ## saving the model
        torch.save(experiment.b_net.state_dict(), os.path.join(dump_dir, "b_net.pt"))