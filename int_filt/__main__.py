import torch
import os

from pathlib import Path

from .experiments import create_experiment

from .utils.config import configuration
from .utils.utils import ensure_reproducibility, dump_config
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
    args = configuration()
    args = vars(args)
    ## dump dir 
    dump_dir = args["dump_dir"]
    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    ## saving configuration
    if args["log"]:
       dump_config(args, os.path.join(dump_dir, "config.json"))
    ## retrieving activations
    args["b_net_activation"] = ACTIVATIONS[args["b_net_activation"]]
    ## retrieving device
    args["device"] = DEVICES[args["device"]]
    ## adding mc configuration
    args["mc_config"] = {"num_samples": args["num_samples"]}
    ## prepare for training drift
    b_net_num_grad_step = args["b_net_num_grad_steps"]
    b_net_optimizer = args["b_net_optimizer"]
    b_net_scheduler = args["b_net_scheduler"]
    b_net_lr = args["b_net_lr"]

    ## setting reproducibility
    random_seed = args["random_seed"]
    ensure_reproducibility(random_seed)
    
    ## creating experiment
    experiment = create_experiment(args)
    ## initializing optimizer and scheduler
    b_net_optimizer = OPTIMIZERS[b_net_optimizer](experiment.b_net.backbone.parameters(), lr = b_net_lr)
    b_net_scheduler = SCHEDULERS[b_net_scheduler]
    if b_net_scheduler is not None:
        b_net_scheduler = b_net_scheduler(b_net_optimizer, b_net_num_grad_step)

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