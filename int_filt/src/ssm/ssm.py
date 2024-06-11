"""
File containing the implementation of template for 
objects representing state space models
"""
import torch

from typing import Optional

from ...utils import ConfigData, InputData, OutputData

class SimSSM(torch.nn.Module):
    """
    Class defining generic methods for implementing state space models simulations
    This is an abstract class and should not be initialized
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary
        """
        super(SimSSM, self).__init__()
        ## initializing attributes
        self.config = config

    def initial_state(self, batch: Optional[InputData] = None) -> OutputData:
        """
        Samples from the model's initial distribution
        """
        raise NotImplementedError

    def state_transition(self, batch: InputData) -> OutputData:
        """
        Samples from the model's state transition
        """
        raise NotImplementedError
    
    def observation(self, batch: InputData) -> OutputData:
        """
        Samples from the model's observations system
        """
        raise NotImplementedError
    
    def simulate(self) -> OutputData:
        """
        Runs the simulation
        """
        raise NotImplementedError
