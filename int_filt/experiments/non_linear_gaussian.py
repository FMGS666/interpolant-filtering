"""
Class for running the experiment on the Ornstein-Uhlenbeck model
"""
import torch

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