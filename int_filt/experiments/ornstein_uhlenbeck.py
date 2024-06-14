"""
Class for running the experiment on the Ornstein-Uhlenbeck model
"""
import torch

from .common import Experiment

from ..utils import ConfigData, OutputData

class OUExperiment(Experiment):
    """
    Class for handling the experiments on the Ornstein-Uhlenbeck model
    """
    def __init__(self, config: ConfigData):
        """
        Constructor with custom config dictionary
        """
        super(OUExperiment, self).__init__(config)
    
    def get_batch(self) -> OutputData:
        """
        Samples a batch from the ssm
        """
        ## sampling from stationary state distribution
        x0 = self.ssm.initial_state().float()
        x1 = self.ssm.initial_state().float()
        xc = x0.float()
        ## sampling from observation model
        y = self.ssm.observation().float()
        ## constructing batch
        batch = {"x0": x0, "x1": x1, "xc": xc, "y": y}
        return batch