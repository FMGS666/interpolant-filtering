"""
This file contains the implementation of the backbone models used in the experiments
"""
import torch

import torch.autograd as autograd

from ...utils import ConfigData, InputData, OutputData, safe_cat, safe_broadcast

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
        self.amortized = self.config["amortized"]
    
    def forward(self, batch: InputData) -> OutputData:
        """
        Forward pass on input batch dictionary
        """
        ## define keys to concatenate
        cat_keys = ["t", "xc", "xt"]
        ## optional amortized learning
        if self.amortized:
            cat_keys.append("y")
        ## define keys for which broadcast is needed
        to_broadcast = ["t"]
        ## safely concatenating batch
        xcat = safe_cat(batch, cat_keys, to_broadcast)
        ## forward pass
        b = self.backbone(xcat)
        return b

class C_Net(torch.nn.Module):
    """
    Class implementing the $c$ model
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary, required keys:
            * `backbone`
        """
        super(C_Net, self).__init__()
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
        c = self.backbone(xcat)
        return c

class C_Net_Wrapper(torch.nn.Module):
    """
    Class implementing a wrapper around the $c$ model
    that allows to automatically satisfy the terminal condition
    """
    def __init__(self, config: ConfigData) -> None:
        super(C_Net_Wrapper, self).__init__()
        ## initializing attributes
        self.config = config
        ## parsing configuration dictionary
        self.model = config["c_net"]
        self.interpolant = config["interpolant"]
        self.observation_model = config["observation_model"]

    def forward(self, batch: InputData) -> OutputData:
        """
        Forward pass on input batch dictionary
        """
        x_clone = batch["xt"].clone().detach().requires_grad_(True)
        ## constructing observation model input dictionary
        dist_batch_dict = {"x": x_clone, "y": batch["y"]}
        log_prob = self.observation_model.log_prob(dist_batch_dict).sum()
        score = autograd.grad(log_prob, x_clone, retain_graph=True)[0]
        ## computing control term
        out_c_net = self.model(batch)
        ## broadcasting to match target shape
        t = safe_broadcast(batch["t"], out_c_net)
        ## computing volatility coefficients
        sigma = self.interpolant.coeffs.sigma(t)
        sigma = safe_broadcast(sigma, score)
        #print(f"{sigma.shape=}, {score.shape=}, {out_c_net.shape}, {t.shape}")
        ## weighted mean to satisfy terminal condition
        out = (1 - t)*out_c_net + t*score*sigma
        return out