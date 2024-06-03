"""
File containing utility objects for handling
probability distributions in the obervation model
"""
import torch
import math

from .types import ConfigData, InputData, OutputData

class ObservationModel:
    """
    Class defining generic methods for implementing observation model
    This is an abstract class and should not be initialized
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom configuration dictionary
        """
        self.config = config

    def log_prob(self, batch: InputData) -> OutputData:
        """
        Return the log_prob of the given batch
        """
        raise NotImplementedError
    
class GaussianObservationModel(ObservationModel):
    """
    Class defining methods for implementing Gaussian observation model
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom configuration dictionary
        """
        super(GaussianObservationModel, self).__init__(config)
        ## parsing configuration dictionary
        self.sigma_y = config["sigma_y"]

    def log_prob(self, batch: InputData) -> OutputData:
        """
        Return the log_prob of the given batch
        """
        ## parsing batch dictionary
        x = batch["input"]
        mu = batch["theta"]
        d = mu.shape[1]
        constants = - 0.5 * d * torch.log(torch.tensor(2 * math.pi, device = x.device)) - 0.5 * d * math.log(self.sigma_y)
        logdensity = torch.squeeze(constants - 0.5 * torch.sum((x - mu)**2, 1) / self.sigma_y)
        return logdensity
    
