"""
File containing the implementation of pendulum state OU model
"""
import torch

import numpy as np

from typing import Optional

from .ssm import SimSSM

from ...utils import ConfigData, InputData, OutputData

class SimOrnsteinUhlenbeck(SimSSM):
    """
    Class implementing a simulation from a Ornstein-Uhlenbeck Model as defined in the 
    Computational Doob's h-transforms for Online Filtering of Discretely Observed
    Diffusions
    Nicolas Chopin 1 Andras Fulop 2 Jeremy Heng Alexandre H. Thiery
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary
        """
        super(SimOrnsteinUhlenbeck, self).__init__(config)
        ## initializing attributes
        self.config = config
        ## parsing configuration dictionary
        self.num_sims = config["num_sims"]
        self.num_dims = config["num_dims"]
        self.sigma_x = config["sigma_x"]
        self.sigma_y = config["sigma_y"]
        self.beta = config["beta"]
    
    def initial_state(self) -> OutputData:
        """
        Samples from the model's initial distribution
        """
        x0 = torch.randn((self.num_sims, self.num_dims))*(self.sigma_x/np.sqrt([2.0*self.beta]))
        return x0
    
    def observation(self) -> OutputData:
        """
        Samples from the model's observation
        """
        state = self.initial_state()
        z = torch.randn_like(state)
        return state + np.sqrt(self.sigma_y)*z