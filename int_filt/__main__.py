import torch
import os

from pathlib import Path

from .experiments import create_experiment

from .utils import (
    configuration, 
    ensure_reproducibility, 
    dump_config,
    dump_tensors, 
    move_batch_to_device
)

ACTIVATIONS = {
    "relu": torch.nn.ReLU()
}

OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adam-w": torch.optim.AdamW
}

SCHEDULERS = {
    "none": None,
    "cosine-annealing": torch.optim.lr_scheduler.CosineAnnealingLR
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
    config["c_net_activation"] = ACTIVATIONS[config["c_net_activation"]]
    config["device"] = DEVICES[config["device"]]
    ## adding mc configuration
    config["mc_config"] = {"num_mc_samples": config["num_mc_samples"]}
    ## setting reproducibility
    ensure_reproducibility(config["random_seed"])
    ## creating experiment
    experiment = create_experiment(config)
    ## dumping particles
    #if config["log_results"]:
    #    experiment.ssm.dump_sim(config["dump_dir"])

    #####################################################################################################################
    ################################################    TRAINING    #####################################################
    #####################################################################################################################
    ## visualizing standardization 
    ## standardization
    batch = experiment.get_batch()
    batch = move_batch_to_device(batch, config["device"])
    print(f"STANDARDIZATION: {experiment.preprocessing.params}")
    print("BEFORE STANDARDIZATION\n")
    for k, v in batch.items():
        print(k, "-> mean: ", v.mean(), ", std: ", v.std(), "shape: ", v.shape)
    batch_pp = experiment.preprocessing(batch)
    print("\nAFTER PREPROCESSING\n")
    for k, v in batch_pp.items():
        print(k, "-> mean: ", v.mean(), ", std: ", v.std(), "shape: ", v.shape)
    ## initializing optimizer and scheduler
    b_net_optimizer = OPTIMIZERS[config["b_net_optimizer"]](experiment.b_net.backbone.parameters(), lr = config["b_net_lr"])
    b_net_scheduler = SCHEDULERS[config["b_net_scheduler"]]
    if b_net_scheduler is not None:
        b_net_scheduler = b_net_scheduler(b_net_optimizer, config["b_net_num_grad_steps"])
    ## initializing optimizer and scheduler for optional control term
    c_net_optimizer = None
    c_net_scheduler = None
    if config["controlled"]:
        c_net_optimizer = OPTIMIZERS[config["c_net_optimizer"]](experiment.c_net.model.backbone.parameters(), lr = config["c_net_lr"])
        c_net_scheduler = SCHEDULERS[config["c_net_scheduler"]]
        if c_net_scheduler is not None:
            c_net_scheduler = c_net_scheduler(c_net_optimizer, config["num_grad_steps"])
    ## constructing optimization config dictionary
    optim_config = {
        "b_net_optimizer": b_net_optimizer,
        "b_net_scheduler": b_net_scheduler,
        "b_net_amortized_optimizer": b_net_amortized_optimizer,
        "b_net_amortized_scheduler": b_net_amortized_scheduler,
        "c_net_optimizer": c_net_optimizer,
        "c_net_scheduler": c_net_scheduler,
        "num_grad_steps": config["num_grad_steps"],
    }
    ## training
    if config["controlled"]:
        train_dict = experiment.train_controlled(optim_config)
    elif config["amortized"]:
        train_dict = experiment.train_amortized(optim_config)
    else:
        train_dict = experiment.train_drift(optim_config)
    ## optional logging
    if config["log_results"]:
        ## saving the model
        torch.save(experiment.b_net.state_dict(), os.path.join(config["dump_dir"], "b_net.pt"))
        if config["controlled"]:
            torch.save(experiment.c_net.state_dict(), os.path.join(config["dump_dir"], "c_net.pt"))
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
    ar_sample_dict = experiment.ar_sample(batch, config = ar_sample_config)
    ## parsing samples dict
    ar_samples = sample_dict["ar_samples"]
    ## displaying the shape of the results
    print(f"{samples.shape=},", end = " ")
    if config["full_out"]:
        trajectory = sample_dict["trajectory"]
        print(f"{trajectory.shape=}")
    if config["log_results"]:
        ## defining target files
        ar_dump_file = os.path.join(config["dump_dir"], "ar_sampling_data")
        train_file = os.path.join(config["dump_dir"], "training_data")
        ## dumping ar sampling and training dictionaries
        dump_tensors(ar_dump_file, ar_sample_dict)
        dump_tensors(train_file, train_dict)
