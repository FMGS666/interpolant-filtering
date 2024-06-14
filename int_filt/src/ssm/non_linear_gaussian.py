"""
File containing the implementation of pendulum state OU model
"""
import torch

import numpy as np

from typing import Optional
from tqdm import tqdm

from .ssm import SimSSM

from ...utils import ConfigData, InputData, OutputData

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
        self.sim = self.simulate()

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
        self.sim = self.simulate()

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
        self.sim = self.simulate()

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
###################################          HYPERBOLIC          #############################################
##############################################################################################################
##############################################################################################################

class SimNLGTanh(SimNonLinearGaussian):
    """
    Class implementing a non-linear gaussian state transition
    with gaussian observation model
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary
        """
        super(SimNLGTanh, self).__init__(config)
        ## running the simulations
        self.sim = self.simulate()

    def non_linearity(self, x: InputData) -> OutputData:
        r"""
        Computes the non-linear update
        $f(x) = x - \delta_t \operatorname{exp} \{x\}$
        """
        update = torch.tanh(self.beta*x)
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
        self.sim = self.simulate()

    def non_linearity(self, x: InputData) -> OutputData:
        r"""
        Computes the non-linear update
        $f(x) = x - \delta_t \operatorname{exp} \{x\}$
        """
        update = torch.exp(-self.beta*x)
        return x + self.step_size*update