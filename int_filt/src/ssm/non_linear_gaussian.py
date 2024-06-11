"""
File containing the implementation of pendulum state OU model
"""
import torch

import numpy as np

from torch.nn.functional import tanh, sigmoid
from typing import Optional
from tqdm import tqdm

from .ssm import SimSSM

from ...utils import ConfigData, InputData, OutputData


class SimNonLinearGaussian(SimSSM):
    ## considered non-linearities
    NON_LINEARITIES = {
        "tanh": tanh,
        "sigmoid": sigmoid
    }
    """
    Class implementing a non-linear gaussian state transition
    with gaussian observation model
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary
        """
        super(SimNonLinearGaussian, self).__init__(config)
        ## parsing configuration dictionary
        self.num_sims = config["num_sims"]
        self.num_dims = config["num_dims"]
        self.num_iters = config["num_iters"]
        self.sigma_x = config["sigma_x"]
        self.sigma_y = config["sigma_y"]
        self.non_linearity = config["non_linearity"]
        ## getting non linearity function
        self.non_linearity_fn = self.NON_LINEARITIES[self.non_linearity]
        ## running the simulations
        self.sim = self.simulate()
    
    def initial_state(self) -> OutputData:
        """
        Samples from the model's initial distribution
        """
        x0 = torch.randn((self.num_sims, self.num_dims))*self.sigma_x
        return x0
    
    def state_transition(self, x: InputData) -> OutputData:
        """
        Samples from the model's state transition
        """
        ## applying non-linearity
        x = self.non_linearity_fn(x)
        ## sampling perturbation
        z = torch.randn_like(x)
        return x + self.sigma_x*z

    def observation(self, x: InputData) -> OutputData:
        """
        Samples from the model's observations system
        """
        ## sampling perturbation
        z = torch.randn_like(x)
        return x + self.sigma_y*z

    def simulate(self) -> OutputData:
        """
        Runs the simulation
        """
        ## allocating memory
        latent_states_store = torch.zeros((self.num_iters, self.num_sims, self.num_dims))
        observation_store = torch.zeros((self.num_iters, self.num_sims, self.num_dims))
        ## sampling first states and observations
        x = self.initial_state()
        y = self.observation(x)
        ## storing states and observations
        latent_states_store[0] = x
        observation_store[0] = y
        ## starting iteration
        for idx in tqdm(range(1, self.num_iters)):
            ## sampling state transition and observation
            x = self.state_transition(x)
            y = self.observation(x)
            ## storing state and observation
            latent_states_store[idx] = x
            observation_store[idx] = y
        ## defining simulation dictionary
        sim = {"latent_states": latent_states_store, "observations": observation_store}
        return sim