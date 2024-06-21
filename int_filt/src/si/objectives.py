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
        self.full_out = self.config["full_out"]
        self.pp_before_interpolant = self.config["pp_before_interpolant"]

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
        ## prepare for mc estimation
        loss_store = torch.zeros((num_mc_samples), device = x0.device)
        if self.full_out:
            drift_store = torch.zeros((num_mc_samples, *x0.shape))
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
            if self.pp_before_interpolant:
                mc_batch = self.preprocessing(mc_batch)
            ## computing interpolant and velocity
            xt = self.interpolant.interpolant(mc_batch)
            rt = self.interpolant.velocity(mc_batch)
            ## augmenting batch
            mc_batch["xt"] = xt
            ## preprocessing batch
            if not self.pp_before_interpolant:
                mc_batch = self.preprocessing(mc_batch)
            ## performing forward pass on the b_net
            bt = self.b_net(mc_batch)
            ## storing the sampled drift
            if self.full_out:
                drift_store[sample_id] = bt.clone().detach()
            ## computing and storing loss
            loss = mse_loss(bt, rt)
            loss_store[sample_id] = loss
        ## aggregating the loss
        loss = torch.mean(loss_store)
        ## constructing output dictionary
        loss_dict = {"loss": loss,} 
        if self.full_out:
            loss_dict["drift_store"] = drift_store
        return loss_dict