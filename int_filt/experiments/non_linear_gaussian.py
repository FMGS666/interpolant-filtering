"""
Class for running the experiment on the Ornstein-Uhlenbeck model
"""
import torch
import math

import numpy as np

from typing import Optional

from .common import Experiment

from ..utils import ConfigData, OutputData

class NLGExperiment(Experiment):
    """
    Class for handling the experiments on the Ornstein-Uhlenbeck model
    """
    def __init__(self, config: ConfigData):
        """
        Constructor with custom config dictionary
        """
        super(NLGExperiment, self).__init__(config)
    
    def get_batch(self, train: bool = True, idx: Optional[int] = None) -> OutputData:
        """
        Samples a batch from the ssm
        """
        ## retrieving necessary data
        num_iters = self.ssm.num_iters
        num_sims = self.ssm.num_sims
        latent_states = self.ssm.train_sim["latent_states"]
        observations = self.ssm.train_sim["observations"]
        ## sampling random time index
        index = torch.randint(num_iters - 1, (1, )).item()
        ## in case idx is passed 
        if idx is not None:
            index = idx
        ## getting current states
        x0 = latent_states[index]
        x1 = latent_states[index + 1]
        y = observations[index + 1]
        xc = x0
        ## constructing the batch
        batch = {"x0": x0.float(), "x1": x1.float(), "xc": xc.float(), "y": y.float()}
        return batch
    
    ## function for evaluating the logpdf of a normal distribution
    def logpdf(self, batch):
        ## parsing batch dictionary and retrieving necessary data
        x = batch["x0"] # shape (num_sims, num_dims)
        samples = batch["samples"] # shape (num_samples , num_sims , num_dims)
        num_dims = self.ssm.num_dims
        num_samples = samples.shape[0] 
        ## computing mean
        mean = self.ssm.non_linearity(x)
        sigma_x = self.ssm.sigma_x
        ## reshaping tensors 
        samples = torch.transpose(samples, 0, 1) # shape (num_sims , num_samples , num_dims)
        mean = torch.unsqueeze(mean, dim = 1) # shape (num_sims , 1 , num_dims)
        mean = mean.repeat([1, num_samples, 1]) # shape (num_sims , num_samples , num_dims)
        ## flattening tensors
        samples = torch.flatten(samples, start_dim = 0, end_dim = 1) # shape (num_sims*num_samples , num_samples , num_dims)
        mean = torch.flatten(mean, start_dim = 0, end_dim = 1) # shape (num_sims*num_samples , num_samples , num_dims)
        ## computing log pdf
        constants = - 0.5 * num_dims * torch.log(torch.tensor(2 * math.pi)) - 0.5 * num_dims * np.log(sigma_x**2)
        logdensity = torch.squeeze(constants - 0.5 * torch.sum((samples - mean)**2, 1) / sigma_x**2)
        ## aggregating results
        logdensity_mean = torch.mean(logdensity)
        logdensity_sum = torch.sum(logdensity)
        ## constructing output dictionary
        logdensity_dict = {"mean": logdensity_mean, "sum": logdensity_sum, "no_reduction": logdensity}
        return logdensity_dict