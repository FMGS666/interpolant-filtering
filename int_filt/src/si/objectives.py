"""
This file contains the implementation of objective functions for working with the Stochastic Interpolant Framework
"""
import torch

from torch.nn.functional import mse_loss
from typing import Optional

from .interpolants import StochasticInterpolant 

from ...utils import ModelData, ConfigData, InputData, OutputData, safe_broadcast

class DriftObjective(torch.nn.Module):
    """
    Class implementing the objective function for learning the drift $b(t, x_0, x)$
    """
    def __init__(self, config: Optional[ConfigData]):
        """
        `config`: dictionary expected keys:
            * b_net 
            * interpolant  
            * mc_config 
        """
        super(DriftObjective, self).__init__()
        ## initializing attributes
        self.config = config
        ## parsing configuration dictionary
        self.b_net = self.config["b_net"]
        self.interpolant = self.config["interpolant"]
        self.mc_config = self.config["mc_config"]
        self.preprocessing = self.config["preprocessing"]

    def forward(self, batch: InputData) -> OutputData:
        """
        Returns the MC estimate of the objective function
        """
        ## parsing batch dictionary
        x0 = batch["x0"]
        x1 = batch["x1"]
        ## retrieve optional conditional context
        xc = x0
        if "xc" in batch.keys() and batch["xc"] is not None:
            xc = batch["xc"]
        ## retrieving observation
        y = batch["y"]
        ## retrieve required data
        num_mc_samples = self.mc_config["num_mc_samples"]
        batch_size = x0.shape[0]
        device = x0.device
        ## allocating memory
        loss_store = torch.zeros((num_mc_samples), device = x0.device)
        ## sampling for mc estimation
        mc_samples = torch.rand((num_mc_samples, batch_size), device = x0.device)
        z_samples = torch.randn((num_mc_samples, *x0.shape), device = device)
        ## start mc sampling
        for sample_id in range(num_mc_samples):
            ## get sampled time indices
            t = mc_samples[sample_id]
            z = z_samples[sample_id]
            ## constructing sample dictionary
            mc_batch = {"t": t, "x0": x0, "x1": x1, "xc": xc, "z": z, "y": y}
            ## preprocessing batch
            mc_batch = self.preprocessing(mc_batch)
            ## computing interpolant and velocity
            xt = self.interpolant.interpolant(mc_batch)
            rt = self.interpolant.velocity(mc_batch)
            ## augmenting batch
            mc_batch["xt"] = xt
            ## performing forward pass on the b_net
            bt = self.b_net(mc_batch)
            ## computing and storing loss
            loss = mse_loss(bt, rt)
            loss_store[sample_id] = loss
        ## aggregating the loss
        loss = torch.mean(loss_store)
        ## constructing output dictionary
        loss_dict = {"loss": loss}
        return loss_dict

class ControlledDriftObjective(torch.nn.Module):
    """
    Class implementing the objective function for learning the drift $b(t, x_0, x)$
    """
    def __init__(self, config: Optional[ConfigData]):
        """
        `config`: dictionary expected keys:
            * b_net 
            * interpolant  
            * mc_config 
        """
        super(ControlledDriftObjective, self).__init__()
        ## initializing attributes
        self.config = config
        ## parsing configuration dictionary
        self.b_net = self.config["b_net"]
        self.c_net = self.config["c_net"]
        self.interpolant = self.config["interpolant"]
        self.mc_config = self.config["mc_config"]
        self.preprocessing = self.config["preprocessing"]

    def forward(self, batch: InputData) -> OutputData:
        """
        Returns the MC estimate of the objective function
        """
        ## parsing batch dictionary
        x0 = batch["x0"]
        x1 = batch["x1"]
        ## retrieve optional conditional context
        xc = x0
        if "xc" in batch.keys() and batch["xc"] is not None:
            xc = batch["xc"]
        ## retrieving observation
        y = batch["y"]
        ## retrieve required data
        num_mc_samples = self.mc_config["num_mc_samples"]
        batch_size = x0.shape[0]
        device = x0.device
        ## allocating memory
        loss_store = torch.zeros((num_mc_samples), device = x0.device)
        b_loss_store = torch.zeros((num_mc_samples), device = x0.device)
        c_loss_store = torch.zeros((num_mc_samples), device = x0.device)
        ## sampling for mc estimation
        mc_samples = torch.rand((num_mc_samples, batch_size), device = x0.device)
        z_samples = torch.randn((num_mc_samples, *x0.shape), device = device)
        ## start mc sampling
        for sample_id in range(num_mc_samples):
            ## get sampled time indices
            t = mc_samples[sample_id]
            z = z_samples[sample_id]
            ## constructing sample dictionary
            mc_batch = {"t": t, "x0": x0, "x1": x1, "xc": xc, "z": z, "y": y}
            ## preprocessing batch
            mc_batch = self.preprocessing(mc_batch)
            ## computing interpolant and velocity
            xt = self.interpolant.interpolant(mc_batch)
            rt = self.interpolant.velocity(mc_batch)
            ## augmenting batch
            mc_batch["xt"] = xt
            ## performing forward pass on the b_net
            bt = self.b_net(mc_batch)
            bt_clone = bt.clone().detach()
            ## computing drift mse loss
            b_loss = mse_loss(bt, rt)
            ## performing forward pass on the c_net
            ct = self.c_net(mc_batch)
            ct_clone = ct.clone().detach()
            ## computing diffusion coefficient
            sigma = self.interpolant.coeffs.sigma(t)
            sigma = safe_broadcast(sigma, ct)
            ## computing controlled loss
            c_loss = mse_loss(ct*sigma, rt) + torch.sum(sigma*ct*bt_clone)
            ## computing full loss
            loss = b_loss + c_loss
            ## storing losses
            loss_store[sample_id] = loss
            b_loss_store[sample_id] = b_loss
            c_loss_store[sample_id] = c_loss
        ## aggregating the loss
        loss = torch.mean(loss_store)
        b_loss = torch.mean(b_loss_store)
        c_loss = torch.mean(c_loss_store)
        ## constructing output dictionary
        loss_dict = {"loss": loss, "b_loss": b_loss, "c_loss": c_loss}
        return loss_dict