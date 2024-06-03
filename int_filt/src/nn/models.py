"""
This file contains the implementation of the backbone models used in the experiments
"""
import torch
import torch.autograd as autograd

from ...utils import ConfigData, ModelData, InputData, OutputData, safe_cat, safe_broadcast

class B_Net(torch.nn.Module):
    """
    Class implementing the $b$ model
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary, required keys:
            * `backbone`
        """
        super(B_Net, self).__init__()
        ## initializing attributes
        self.config = config
        ## parsing configuration dictionary
        self.backbone = self.config["backbone"]
    
    def forward(self, batch: InputData) -> OutputData:
        """
        Forward pass on input batch dictionary
        """
        ## define keys to concatenate
        cat_keys = ["t", "xc", "xt", "y"]
        ## define keys for which broadcast is needed
        to_broadcast = ["t"]
        ## safely concatenating batch
        xcat = safe_cat(batch, cat_keys, to_broadcast)
        ## forward pass
        b = self.backbone(xcat)
        return b