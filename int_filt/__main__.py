import torch
import os

from pathlib import Path

from .experiments import create_experiment

from .utils import (
    configuration, 
    ensure_reproducibility, 
    dump_config,
    move_batch_to_device
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
    print(config)
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
    ## dumping particles
    if config["log_results"]:
        experiment.ssm.dump_sim(config["dump_dir"])

    #####################################################################################################################
    ################################################    TRAINING    #####################################################
    #####################################################################################################################
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
        torch.save(experiment.b_net.state_dict(), os.path.join(config["dump_dir"], "b_net.pt"))

    #####################################################################################################################
    ###############################################    GENERATION    ####################################################
    #####################################################################################################################
    ## constructing autoregressive sampling config dictionary
    ar_sample_config = {
        "num_time_steps": config["num_time_steps"],
        "num_ar_steps": config["num_ar_steps"],
        "initial_time_step": config["initial_time_step"],
        "ar_sample_train": config["ar_sample_train"],
    }
    ## sampling from model
    sample_dict = experiment.ar_sample(batch, config = ar_sample_config)
    ## parsing samples dict
    ar_samples = sample_dict["ar_samples"]
    ## displaying the shape of the results
    print(f"{samples.shape=},", end = " ")
    if config["full_out"]:
        trajectory = sample_dict["trajectory"]
        drift = sample_dict["drift"]
        diffusion = sample_dict["diffusion"]
        print(f"{trajectory.shape=}, {drift.shape=}, {diffusion.shape=}")