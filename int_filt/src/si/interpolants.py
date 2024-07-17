"""
This file contains the implementation of customizable interpolant function in the context of the stochastic interpolant framework
"""

import torch

from typing import Optional

from ...utils import ConfigData, InputData, OutputData, safe_broadcast

class StochasticInterpolant(torch.nn.Module):
    """
    Class defining generic methods for implementing stochastic interpolants
    This is an abstract class and should not be initialized
    """
    def __init__(self, config: Optional[ConfigData] = None) -> None:
        """
        Constructor with custom config dictionary
        """
        super(StochasticInterpolant, self).__init__()
        ## initializing attributes
        self.config = config

    """
    Methods for computing the interpolant
    """
    def phi(self, batch: InputData) -> OutputData:
        r"""
        Computes the deterministic term $\phi(t, x_0, x_1)$ of the interpolant
        """
        raise NotImplementedError

    def gamma(self, batch: InputData) -> OutputData:
        r"""
        Computes the stochastic term $\gamma(t, x_0, x_1)$ of the interpolant
        """
        raise NotImplementedError

    def interpolant(self, batch: InputData) -> OutputData:
        r"""
        Computes the stochastic interpolant $x_t = \phi(t, x_0, x_1) + \gamma(t, x_0, x_1) z$
        """
        z = batch["z"]
        phi = self.phi(batch)
        gamma = self.gamma(batch)
        return phi + gamma*z

    """
    Methods for computing the velocity
    """
    def phi_dot(self, batch: InputData) -> OutputData:
        r"""
        Computes the velocity of deterministic term $\dot{\phi}(t, x_0, x_1)$
        """
        raise NotImplementedError

    def gamma_dot(self, batch: InputData) -> OutputData:
        r"""
        Computes the velocity of stochastic term $\dot{\gamma}(t, x_0, x_1)$
        """
        raise NotImplementedError

    def velocity(self, batch: InputData) -> OutputData:
        """
        Computes the interpolant velocity
        """
        z = batch["z"]
        phi_dot = self.phi_dot(batch)
        gamma_dot = self.gamma_dot(batch)
        return phi_dot + gamma_dot*z

class PFFPInterpolant(StochasticInterpolant):
    """
    Class implementing the stochastic interpolant in the paper 
    "Probabilistic Forecasting with Stochastic Interpolants and Follmer Processes" 
    """
    def __init__(self, config: Optional[ConfigData] = None) -> None:
        """
        `config`: dictionary expected keys:
            * alpha  and alpha_dot
            * beta   and beta_dot
            * sigma  and sigma_dot
        """
        super(PFFPInterpolant, self).__init__(config)
        ## parsing configuration dictionary
        self.coeffs = self.config["coeffs"]

    """
    Methods for computing the interpolant
    """
    def phi(self, batch: InputData) -> OutputData:
        r"""
        Computes the deterministic term $\phi(t, x_0, x_1)$ of the interpolant
        """
        ## parsing batch dictionary
        t = batch["t"]
        x0 = batch["x0"]
        x1 = batch["x1"]
        ## computing deterministic interpolant 
        alpha = self.coeffs.alpha(t)
        beta = self.coeffs.beta(t)
        ## reshaping coefficients
        alpha = safe_broadcast(alpha, x0)
        beta = safe_broadcast(beta, x0)
        return alpha*x0 + beta*x1

    def gamma(self, batch: InputData) -> OutputData:
        r"""
        Computes the stochastic term $\gamma(t, x_0, x_1)$ of the interpolant
        """
        ## parsing batch dictionary
        t = batch["t"]
        z = batch["z"]
        ## computing stochastic interpolant coefficient
        sigma = self.coeffs.sigma(t)
        ## reshaping coefficients
        t = safe_broadcast(t, z)
        sigma = safe_broadcast(sigma, z)
        return torch.sqrt(t)*sigma

    """
    Methods for computing the velocity
    """
    def phi_dot(self, batch: InputData) -> OutputData:
        r"""
        Computes the velocity of deterministic term $\dot{\phi}(t, x_0, x_1)$
        """
        ## parsing batch dictionary
        t = batch["t"]
        x0 = batch["x0"]
        x1 = batch["x1"]
        ## computing deterministic velocity
        alpha_dot = self.coeffs.alpha_dot(t)
        beta_dot = self.coeffs.beta_dot(t)
        ## reshaping coefficients
        alpha_dot = safe_broadcast(alpha_dot, x0)
        beta_dot = safe_broadcast(beta_dot, x0)
        return alpha_dot*x0 + beta_dot*x1

    def gamma_dot(self, batch: InputData) -> OutputData:
        r"""
        Computes the velocity of stochastic term $\dot{\gamma}(t, x_0, x_1)$
        """
        ## parsing batch dictionary
        t = batch["t"]
        z = batch["z"]
        ## computing stochastic velocity coefficient
        sigma_dot = self.coeffs.sigma_dot(t)
        ## reshaping coefficients
        t = safe_broadcast(t, z)
        sigma_dot = safe_broadcast(sigma_dot, z)
        return torch.sqrt(t)*sigma_dot