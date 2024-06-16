"""
File containing the implementation of utility functions 
for proper allocation of stochastic interpolants
"""
import torch

from .interpolants import StochasticInterpolant, PFFPInterpolant
from .pffp_coefficients import PFFPInterpolantCoefficients_v0, PFFPInterpolantCoefficients_v1

from ...utils import ConfigData

def create_interpolant(config: ConfigData) -> StochasticInterpolant:
    if config["method"] == "pffp_v0":
        ## setting epsilon to 1 by defauls
        epsilon = config["epsilon"]
        ## initializing the interpolant coefficients
        coeffs_config = {"epsilon": epsilon}
        coeffs = PFFPInterpolantCoefficients_v0(coeffs_config)
        ## initializing the interpolant
        interp_config = {"coeffs": coeffs}
        interpolant = PFFPInterpolant(interp_config)
    if config["method"] == "pffp_v1":
        ## setting epsilon to 1 by defauls
        epsilon = config["epsilon"]
        ## initializing the interpolant coefficients
        coeffs_config = {"epsilon": epsilon}
        coeffs = PFFPInterpolantCoefficients_v1(coeffs_config)
        ## initializing the interpolant
        interp_config = {"coeffs": coeffs}
        interpolant = PFFPInterpolant(interp_config)
    ## Additional interpolants to be added here
    return interpolant
     