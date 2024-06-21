"""
File containing the implementation of pendulum state OU model
"""
import torch
import os

import numpy as np

from typing import Optional
from tqdm import tqdm

from .ssm import SimSSM

from ...utils import (
    ConfigData, 
    InputData, 
    OutputData,
    PathData, 
    dump_tensors
)

class SimNonLinearGaussian(SimSSM):
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
        self.beta = config["beta"]
        self.step_size = config["step_size"]
        self.num_burn_in_steps = config["num_burn_in_steps"]
    
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
        x = self.non_linearity(x)
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
        ## defining iterator
        num_iterations = self.num_iters + self.num_burn_in_steps
        iterator = tqdm(range(1, num_iterations))
        ## starting iteration
        for idx in iterator:
            ## sampling state transition and observation
            x = self.state_transition(x)
            y = self.observation(x)
            ## storing state and observation after burn in period
            if idx >= self.num_burn_in_steps:
                idx = idx - self.num_burn_in_steps
                latent_states_store[idx] = x
                observation_store[idx] = y
        ## defining simulation dictionary
        sim = {"latent_states": latent_states_store, "observations": observation_store}
        return sim
    
    def dump_sim(self, target_dir: PathData) -> None:
        """
        Dumps the simulation to target directory
        """
        ## defining target files
        train_file = os.path.join(target_dir, "train.npz")
        test_file = os.path.join(target_dir, "test.npz")
        ## saving train simulation
        dump_tensors(train_file, self.train_sim)
        ## saving test simulation
        dump_tensors(test_file, self.test_sim)

##############################################################################################################
##############################################################################################################
################################### DEFINING THE NON LINEARITIES #############################################
###################################         TRIGONOMETRIC        #############################################
##############################################################################################################
##############################################################################################################

class SimNLGCos(SimNonLinearGaussian):
    """
    Class implementing a non-linear gaussian state transition
    with gaussian observation model
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary
        """
        super(SimNLGCos, self).__init__(config)
        ## running the simulations
        self.train_sim = self.simulate()
        self.test_sim = self.simulate()

    def non_linearity(self, x: InputData) -> OutputData:
        r"""
        Computes the non-linear update
        $f(x) = x - \delta_t \operatorname{cos}(x)$
        """
        update = torch.cos(self.beta*x)
        return x + self.step_size*update

class SimNLGSin(SimNonLinearGaussian):
    """
    Class implementing a non-linear gaussian state transition
    with gaussian observation model
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary
        """
        super(SimNLGSin, self).__init__(config)
        ## running the simulations
        self.train_sim = self.simulate()
        self.test_sim = self.simulate()

    def non_linearity(self, x: InputData) -> OutputData:
        r"""
        Computes the non-linear update
        $f(x) = x - \delta_t \operatorname{exp} \{x\}$
        """
        update = torch.sin(self.beta*x)
        return x + self.step_size*update

class SimNLGTan(SimNonLinearGaussian):
    """
    Class implementing a non-linear gaussian state transition
    with gaussian observation model
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary
        """
        super(SimNLGTan, self).__init__(config)
        ## running the simulations
        self.train_sim = self.simulate()
        self.test_sim = self.simulate()

    def non_linearity(self, x: InputData) -> OutputData:
        r"""
        Computes the non-linear update
        $f(x) = x - \delta_t \operatorname{exp} \{x\}$
        """
        update = torch.tan(self.beta*x)
        return x + self.step_size*update

##############################################################################################################
##############################################################################################################
################################### DEFINING THE NON LINEARITIES #############################################
###################################          EXPONENTIAL         #############################################
##############################################################################################################
##############################################################################################################

class SimNLGExp(SimNonLinearGaussian):
    """
    Class implementing a non-linear gaussian state transition
    with gaussian observation model
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary
        """
        super(SimNLGExp, self).__init__(config)
        ## running the simulations
        self.train_sim = self.simulate()
        self.test_sim = self.simulate()

    def non_linearity(self, x: InputData) -> OutputData:
        r"""
        Computes the non-linear update
        $f(x) = x - \delta_t \operatorname{exp} \{x\}$
        """
        update = torch.exp(-self.beta*x)
        return x + self.step_size*update