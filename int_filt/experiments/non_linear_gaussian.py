"""
Script for running the experiment on the Ornstein-Uhlenbeck model
"""
import torch

import numpy as np

from tqdm import tqdm
from typing import Optional

from .common import Experiment

from ..src import DriftObjective

from ..utils import ConfigData, InputData, OutputData, move_batch_to_device, construct_time_discretization

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
        ## computing standardization mean over 
        ## time and simulation dimensions
        latent_states = self.ssm.sim["latent_states"]
        observations = self.ssm.sim["observations"]
        self.mean_x = torch.mean(latent_states, dim = (0, 1))
        self.mean_y = torch.mean(observations, dim = (0, 1))
    
    def get_batch(self, N: Optional[int] = None) -> OutputData:
        """
        Samples a batch from the ssm
        """
        ## retrieving necessary data
        num_iters = self.ssm.num_iters
        num_sims = self.ssm.num_sims
        latent_states = self.ssm.sim["latent_states"]
        observations = self.ssm.sim["observations"]
        ## setting optional number of observations
        if N is not None:
            num_sims = N
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
        latent_state_keys = ["x0", "x1", "xc", "xt"]
        observation_keys = ["y"]
        skip_keys = ["t"]
        ## retrieving necessary data
        sigma_x = self.ssm.sigma_x
        sigma_y = self.ssm.sigma_y
        ## normalizing batch
        batch_copy = dict()
        for key, tensor in batch.items():
            device = tensor.device
            ## computing target std
            if key in latent_state_keys:
                mean = self.mean_x.to(device)
                #std = sigma_x * torch.ones_like(tensor)
            elif key in observation_keys:
                mean = self.mean_y.to(device)
                #std = np.sqrt(sigma_x**2  + sigma_y**2) * torch.ones_like(tensor)
            elif key in skip_keys:
                batch_copy[key] = tensor
                continue
            std = torch.std(tensor).to(device)
            ## scaling tensor
            tensor = (tensor - mean) / std
            #tensor = (tensor - mean)
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
            "mc_config": self.mc_config,
            "preprocess_batch": self.standardize
        }
        Lb = DriftObjective(Lb_config)
        ## defining store loss
        loss_history = torch.zeros((num_grad_steps))
        ## defining iterator
        iterator = tqdm(range(num_grad_steps))
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
            iterator.set_description(f"Grad Step {grad_step + 1}/{num_grad_steps}, MSELoss: {loss_value}")
            iterator.update()
            if grad_step % self.logging_step == 0:
                self.writer.add_scalar("train/drift_loss", loss_value, grad_step)
        return loss_history
    
    def simulate_sde(self, batch: InputData, sample_config: ConfigData) -> OutputData:
        r"""
        Simulates the SDE $dX_t = b(t, X_t)dt + \sigma_tdB_t$
        """
        ## parsing configuration dictionary
        num_time_steps = sample_config["num_time_steps"]
        ## constructing time discretization
        time, stepsizes = construct_time_discretization(num_time_steps, self.device)
        ## augmenting batch
        batch["t"] = time[0]
        batch["xt"] = batch["x0"]
        ## computing drift
        drift = stepsizes[0]*self.b_net(batch)
        ## sampling noise 
        eta = torch.randn_like(drift)
        ## computing diffusion
        diffusion = self.interpolant.coeffs.sigma(time[0])*torch.sqrt(stepsizes[0])*eta
        # updating state and current batch
        x = batch["x0"]
        x = x + drift + diffusion
        # iterating over each step of the euler discretization
        for n in range(1, num_time_steps):
            # gettng the stepsize
            delta_t = stepsizes[n]
            t = time[n]
            ## updating current batch
            batch["xt"] = x
            batch["t"] = time[0]
            ## standardizing batch
            batch = self.standardize(batch)
            # computing adjusted drift
            drift = delta_t*self.b_net(batch)
            # sampling noise
            eta = torch.randn_like(drift)
            # computing diffusion term
            diffusion = self.interpolant.coeffs.sigma(t)*torch.sqrt(delta_t)*eta
            # euler step
            x = x + drift + diffusion
        return x
    
    def sample(self, batch: InputData, sample_config: ConfigData) -> OutputData:
        r"""
        Samples  from the model by simulating the SDE $dX_t = b(t, X_t)dt + \sigma_tdB_t$
        """
        ## parsing configuration dictionary
        num_samples = sample_config["num_samples"]
        ## retrieving necessary data
        num_sims = self.ssm.num_sims
        num_dims = self.ssm.num_dims
        ## allocating memory
        samples_store = torch.zeros(num_samples, num_sims, num_dims)
        ## iterating over each sample
        for sample_id in tqdm(range(num_samples)):
            X1 = self.simulate_sde(batch, sample_config)
            samples_store[sample_id] = X1.detach().cpu()
        return samples_store
