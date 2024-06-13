"""
File containing common code for running experiment
"""
import torch

from typing import Optional

from ..utils import ConfigData, InputData, OutputData

class Experiment:
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary
        """
        ## initializing attributes
        self.config = config

    def get_batch(self) -> OutputData:
        """
        Samples a batch from the ssm
        """
        raise NotImplementedError

    def standardize(self, batch: InputData) -> OutputData:
        """
        standardizes a batch of data
        """
        raise NotImplementedError

    def train(self, optim_config: Optional[ConfigData] = None) -> OutputData:
        """
        Trains the $b$ model
        """
        raise NotImplementedError

    def simulate_sde(self, batch: InputData, sample_config: Optional[ConfigData] = None) -> OutputData:
        r"""
        Simulates the SDE $dX_t = b(t, X_t)dt + \sigma_tdB_t$
        """
        raise NotImplementedError

    def sample(self, batch: InputData, sample_config: Optional[ConfigData] = None) -> OutputData:
        r"""
        Samples  from the model by simulating the SDE $dX_t = b(t, X_t)dt + \sigma_tdB_t$
        """
        raise NotImplementedError

    def FA_APF(self, filter_conf: Optional[ConfigData] = None) -> OutputData:
        """
        Runs Fully Augmented Auxiliary Particle Filter
        """
        raise NotImplementedError