""""
This file contains the implementation of the interpolant 
coefficients used in the PFFP Paper
"""
import torch

from ...utils import ConfigData, InputData, OutputData

"""
Classes implementing the base interpolator used
for the experiments in the PFFP paper
"""
class PFFPInterpolantCoefficients_v0:
    r"""
    PFFP Interpolator v0:
        * $\alpha_s = 1 - s$
        * $\beta_s = s^2$
        * $\sigma_s = \epsilon(1 - s)$ 
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary
        """
        ## initializing attributes
        self.config = config
        ## parsing configuration dictionary
        self.epsilon = config["epsilon"]
    
    """
    Methods for computing the interpolant coefficients
    """
    def alpha(self, t: InputData) -> OutputData:
        ones = torch.ones_like(t)
        return ones - t

    def beta(self, t: InputData) -> OutputData:
        return torch.square(t)

    def sigma(self, t: InputData) -> OutputData:
        ones = torch.ones_like(t)
        return self.epsilon*(ones - t)

    """
    Methods for computing the velocity coefficients
    """
    def alpha_dot(self, t: InputData) -> OutputData:
        ones = torch.ones_like(t)
        return -ones
    
    def beta_dot(self, t: InputData) -> OutputData:
        return 2*t

    def sigma_dot(self, t: InputData) -> OutputData:
        ones = torch.ones_like(t)
        return -self.epsilon*ones 


class PFFPInterpolantCoefficients_v1:
    r"""
    PFFP Interpolator v1:
        * $\alpha_s = 1 - s$
        * $\beta_s = s$
        * $\sigma_s = \epsilon(1 - s)$ 
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary
        """
        ## initializing attributes
        self.config = config
        ## parsing configuration dictionary
        self.epsilon = config["epsilon"]
    
    """
    Methods for computing the interpolant coefficients
    """
    def alpha(self, t: InputData) -> OutputData:
        ones = torch.ones_like(t)
        return ones - t

    def beta(self, t: InputData) -> OutputData:
        return t

    def sigma(self, t: InputData) -> OutputData:
        ones = torch.ones_like(t)
        return self.epsilon*(ones - t)

    """
    Methods for computing the velocity coefficients
    """
    def alpha_dot(self, t: InputData) -> OutputData:
        ones = torch.ones_like(t)
        return -ones

    def beta_dot(self, t: InputData) -> OutputData:
        ones = torch.ones_like(t)
        return ones

    def sigma_dot(self, t: InputData) -> OutputData:
        ones = torch.ones_like(t)
        return -self.epsilon*ones