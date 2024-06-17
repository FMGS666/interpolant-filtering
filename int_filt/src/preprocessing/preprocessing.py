"""
File containing the implementation of standardization preprocessing
"""
import torch 

from typing import Optional

from ...utils import ConfigData, InputData, OutputData, standardize

class IdentityPreproc(torch.nn.Module):
    """
    Class implementing the identity preprocessing
    """
    def __init__(self, config: Optional[ConfigData] = None) -> None:
        """
        Constructor with custom config dictionary
        """
        super(IdentityPreproc, self).__init__()
        ## initializing attributes
        self.config = config
    
    def forward(self, batch: InputData) -> OutputData:
        """
        Performs preprocessing on a batch of data
        """
        return batch

class StandardizeSim(torch.nn.Module):
    """
    Class implementing the standardization with parameters 
    inferred from the whole history of all simulations
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary
        """
        super(StandardizeSim, self).__init__()
        ## parsing configuration dictionary
        self.ssm = config["ssm"]
        ## computing params
        self.params = self.get_params()

    def get_params(self) -> OutputData:
        """
        Computes the mean and standard deviation for states and observations 
        """
        ## retrieving latent states and observations
        latent_states = self.ssm.train_sim["latent_states"]
        observations = self.ssm.train_sim["observations"]
        ## computing mean of latent states and observations
        mean_x = torch.mean(latent_states, dim = 0)
        mean_y = torch.mean(observations, dim = 0)
        ## computing std of latent states and observations
        std_x = torch.std(latent_states, dim = 0)
        std_y = torch.std(observations, dim = 0)
        ## constructing output dictionary
        params = {"mean_x": mean_x, "mean_y": mean_y, "std_x": std_x, "std_y": std_y}
        return params

    def forward(self, batch: InputData) -> OutputData:
        """
        Performs preprocessing on a batch of data
        """
        ## defining keys for latent states and observations
        latent_state_keys = ["x0", "x1", "xc", "xt"]
        observation_keys = ["y"]
        ## normalizing batch
        batch_copy = dict()
        for key, tensor in batch.items():
            ## normalizing with appropriate params
            if key in latent_state_keys:
                mean = self.params["mean_x"]
                std = self.params["std_x"]
                tensor = standardize(tensor, mean, std)
            if key in observation_keys:
                mean = self.params["mean_y"]
                std = self.params["std_y"]
                tensor = standardize(tensor, mean, std)
            ## copying tensor
            batch_copy[key] = tensor
        return batch_copy
