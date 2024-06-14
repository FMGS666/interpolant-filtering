"""
Class for running the experiment on the Ornstein-Uhlenbeck model
"""
import torch

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
        ## parsing configuration dictionary
        self.interpolant = self.config["interpolant"]
        self.b_net = self.config["b_net"]
        self.ssm = self.config["ssm"]
        self.writer = self.config["writer"]
        self.mc_config = self.config["mc_config"]
        self.device = self.config["device"]
        self.preprocessing = self.config["preprocessing"]
        self.logging_step = 5
    
    def get_batch(self) -> OutputData:
        """
        Samples a batch from the ssm
        """
        ## retrieving necessary data
        num_iters = self.ssm.num_iters
        num_sims = self.ssm.num_sims
        latent_states = self.ssm.sim["latent_states"]
        observations = self.ssm.sim["observations"]
        ## sampling random indices
        indices = torch.randint(0, num_iters - 1, (num_sims, ))
        ## getting current states
        x0 = torch.diagonal(latent_states[indices]).T
        x1 = torch.diagonal(latent_states[indices + 1]).T
        y = torch.diagonal(observations[indices + 1]).T
        xc = x0
        ## constructing the batch
        batch = {"x0": x0.float(), "x1": x1.float(), "xc": xc.float(), "y": y.float()}
        return batch