"""
Script for running the experiment on the Ornstein-Uhlenbeck model
"""
import torch

import numpy as np

from .common import Experiment

from ..src import DriftObjective

from ..utils import ConfigData, InputData, OutputData, move_batch_to_device

class NLGExperiment(Experiment):
    """
    Class for handling the experiments on the Ornstein-Uhlenbeck model
    """
    def __init__(self, config: ConfigData):
        """
        Constructor with custom config dictionary
        """
        super(NLGExperiment, self).__init__(config)
        ## parsing configuration dictionary
        self.interpolant = self.config["interpolant"]
        self.b_net = self.config["b_net"]
        self.ssm = self.config["ssm"]
        self.writer = self.config["writer"]
        self.mc_config = self.config["mc_config"]
        self.device = self.config["device"]
        self.logging_step = 5
    
    def get_batch(self) -> OutputData:
        """
        Samples a batch from the ssm
        """
        ## retrieving required data
        num_iters = self.ssm.num_iters
        num_sims = self.ssm.num_sims
        latent_states = self.ssm.sim["latent_states"]
        observations = self.ssm.sim["observations"]
        ## sampling random indices
        indices = torch.randint(0, num_iters - 1, (num_sims, ))
        ## getting current states
        x0 = torch.diagonal(latent_states[indices]).T
        x1 = torch.diagonal(latent_states[indices + 1]).T
        y = torch.diagonal(observations[indices + 1]).T
        xc = x0
        ## constructing the batch
        batch = {"x0": x0.float(), "x1": x1.float(), "xc": xc.float(), "y": y.float()}
        return batch

    def standardize(self, batch: InputData) -> OutputData:
        """
        standardizes a batch of data
        """
        ## defining keys for latent states
        latent_state_keys = ["x0", "x1", "xc"]
        observation_keys = ["y"]
        ## retrieving necessary data
        sigma_x = self.ssm.sigma_x
        sigma_y = self.ssm.sigma_y
        ## normalizing batch
        batch_copy = dict()
        for key, tensor in batch.items():
            ## computing target std
            if key in latent_state_keys:
                mean = torch.mean(tensor, dim = 0, keepdim = True)
                std = sigma_x * torch.ones_like(tensor)
            elif key in observation_keys:
                mean = torch.mean(tensor, dim = 0, keepdim = True)
                std = np.sqrt(sigma_x**2  + sigma_y**2) * torch.ones_like(tensor)
            ## scaling tensor
            tensor = (tensor - mean) / std
            batch_copy[key] = tensor
        return batch_copy

    def train(self, optim_config: ConfigData) -> OutputData:
        """
        Trains the $b$ model
        """
        ## parsing configuration dictionary
        num_grad_steps = optim_config["num_grad_steps"]
        optimizer = optim_config["optimizer"]
        scheduler = optim_config["scheduler"]
        ## initializing objective function
        Lb_config = {
            "b_net": self.b_net, 
            "interpolant": self.interpolant, 
            "mc_config": self.mc_config
        }
        Lb = DriftObjective(Lb_config)
        ## defining store loss
        loss_history = torch.zeros((num_grad_steps))
        ## starting optimization
        for grad_step in range(num_grad_steps):
            ## preparingg batch
            batch = self.get_batch()
            #batch = self.standardize(batch)
            batch = move_batch_to_device(batch, self.device)
            ## estimating loss
            loss = Lb.forward(batch)
            loss_value = loss.item()
            ## optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ## storing loss
            loss_history[grad_step] = loss_value
            ## logging
            if grad_step % self.logging_step == 0:
                self.writer.add_scalar("train/drift_loss", loss_value, grad_step)
                print(f"Grad step: {grad_step}, Velocity Field MSE Loss: {loss_value}", flush = True)
        return loss_history