"""
File containing the test for the `surr_si.[interpolants, pffp_coefficients]`  module
"""
import torch

from .common import Test

from ..src import create_interpolant
from ..utils import InputData

######################################################
################ Surr SI Interpolant #################
######################################################
class SurrSIInterpolantTest(Test):
    def setUp(self) -> None:
        ## initializing pffp-v0 interpolant
        v0_config = {"method": "pffp_v0"}
        self.interpolant_v0 = create_interpolant(v0_config)

        ## initializing pffp-v0 interpolant
        v1_config = {"method": "pffp_v1"}
        self.interpolant_v1 = create_interpolant(v1_config)

        ## defining expected size
        self.target_shape = torch.Size((16, 2, 2))
        
        ## defining testing at times 0.0, 0.5 and 1.0
        self.t0 = torch.zeros(self.target_shape)
        self.t05 = torch.zeros(self.target_shape) + 0.5
        self.t075 = torch.zeros(self.target_shape) + 0.75
        self.t1 = torch.ones(self.target_shape)

    ######################################################
    ############### PFFP-v0 Interpolant ##################
    ######################################################
    def test_surr_si_intrpl_pffp_v0_coeffs_alpha(self) -> None:
        ## computing alpha coefficient
        alpha0 = self.interpolant_v0.coeffs.alpha(self.t0)
        alpha05 = self.interpolant_v0.coeffs.alpha(self.t05)
        alpha075 = self.interpolant_v0.coeffs.alpha(self.t075)
        alpha1 = self.interpolant_v0.coeffs.alpha(self.t1)
    
        ## defining the target values
        alpha0_target = torch.ones(self.target_shape)
        alpha05_target = torch.zeros(self.target_shape) + 0.5
        alpha075_target = torch.zeros(self.target_shape) + 0.25
        alpha1_target = torch.zeros(self.target_shape)

        ## testing shape 
        self.check_shape(alpha0, self.target_shape)
        self.check_shape(alpha05, self.target_shape)
        self.check_shape(alpha075, self.target_shape)
        self.check_shape(alpha1, self.target_shape)
        
        ## testing target values
        self.check_equal(alpha0, alpha0_target)
        self.check_equal(alpha05, alpha05_target)
        self.check_equal(alpha075, alpha075_target)
        self.check_equal(alpha1, alpha1_target)

    def test_surr_si_intrpl_pffp_v0_coeffs_beta(self) -> None:
        ## computing alpha coefficient
        beta0 = self.interpolant_v0.coeffs.beta(self.t0)
        beta05 = self.interpolant_v0.coeffs.beta(self.t05)
        beta075 = self.interpolant_v0.coeffs.beta(self.t075)
        beta1 = self.interpolant_v0.coeffs.beta(self.t1)
    
        ## defining the target values
        beta0_target = torch.zeros(self.target_shape)
        beta05_target = torch.zeros(self.target_shape) + 0.25
        beta075_target = torch.zeros(self.target_shape) + 0.5625
        beta1_target = torch.ones(self.target_shape)

        ## testing shape 
        self.check_shape(beta0, self.target_shape)
        self.check_shape(beta05, self.target_shape)
        self.check_shape(beta075, self.target_shape)
        self.check_shape(beta1, self.target_shape)
        
        ## testing target values
        self.check_equal(beta0, beta0_target)
        self.check_equal(beta05, beta05_target)
        self.check_equal(beta075, beta075_target)
        self.check_equal(beta1, beta1_target)
    
    def test_surr_si_intrpl_pffp_v0_coeffs_sigma(self) -> None:
        ## computing alpha coefficient
        sigma0 = self.interpolant_v0.coeffs.sigma(self.t0)
        sigma05 = self.interpolant_v0.coeffs.sigma(self.t05)
        sigma075 = self.interpolant_v0.coeffs.sigma(self.t075)
        sigma1 = self.interpolant_v0.coeffs.sigma(self.t1)
    
        ## defining the target values
        sigma0_target = torch.ones(self.target_shape)
        sigma05_target = torch.zeros(self.target_shape) + 0.5
        sigma075_target = torch.zeros(self.target_shape) + 0.25
        sigma1_target = torch.zeros(self.target_shape)

        ## testing shape 
        self.check_shape(sigma0, self.target_shape)
        self.check_shape(sigma05, self.target_shape)
        self.check_shape(sigma075, self.target_shape)
        self.check_shape(sigma1, self.target_shape)
        
        ## testing target values
        self.check_equal(sigma0, sigma0_target)
        self.check_equal(sigma05, sigma05_target)
        self.check_equal(sigma075, sigma075_target)
        self.check_equal(sigma1, sigma1_target)
    
    ######################################################
    ################ PFFP-v0 Velocity ####################
    ######################################################
    def test_surr_si_intrpl_pffp_v0_coeffs_alpha_dot(self) -> None:
        ## computing alpha coefficient
        alpha_dot0 = self.interpolant_v0.coeffs.alpha_dot(self.t0)
        alpha_dot05 = self.interpolant_v0.coeffs.alpha_dot(self.t05)
        alpha_dot075 = self.interpolant_v0.coeffs.alpha_dot(self.t075)
        alpha_dot1 = self.interpolant_v0.coeffs.alpha_dot(self.t1)
    
        ## defining the target values
        alpha_dot0_target = -torch.ones(self.target_shape)
        alpha_dot05_target = -torch.zeros(self.target_shape)
        alpha_dot075_target = -torch.zeros(self.target_shape)
        alpha1_target = -torch.zeros(self.target_shape)

        ## testing shape 
        self.check_shape(alpha_dot0, self.target_shape)
        self.check_shape(alpha_dot05, self.target_shape)
        self.check_shape(alpha_dot075, self.target_shape)
        self.check_shape(alpha_dot1, self.target_shape)

    def test_surr_si_intrpl_pffp_v0_coeffs_beta_dot(self) -> None:
        ## computing alpha coefficient
        beta_dot0 = self.interpolant_v0.coeffs.alpha_dot(self.t0)
        beta_dot05 = self.interpolant_v0.coeffs.alpha_dot(self.t05)
        beta_dot075 = self.interpolant_v0.coeffs.alpha_dot(self.t075)
        beta_dot1 = self.interpolant_v0.coeffs.alpha_dot(self.t1)
    
        ## defining the target values
        beta_dot0_target = 2*self.t0
        beta_dot05_target = 2*self.t05
        beta_dot075_target = 2*self.t075
        beta1_target = 2*self.t1

        ## testing shape 
        self.check_shape(beta_dot0, self.target_shape)
        self.check_shape(beta_dot05, self.target_shape)
        self.check_shape(beta_dot075, self.target_shape)
        self.check_shape(beta_dot1, self.target_shape)
    
    def test_surr_si_intrpl_pffp_v0_coeffs_sigma_dot(self) -> None:
        ## computing alpha coefficient
        sigma_dot0 = self.interpolant_v0.coeffs.sigma_dot(self.t0)
        sigma_dot05 = self.interpolant_v0.coeffs.sigma_dot(self.t05)
        sigma_dot075 = self.interpolant_v0.coeffs.sigma_dot(self.t075)
        sigma_dot1 = self.interpolant_v0.coeffs.sigma_dot(self.t1)
    
        ## defining the target values
        sigma_dot0_target = -torch.ones(self.target_shape)
        sigma_dot05_target = -torch.zeros(self.target_shape)
        sigma_dot075_target = -torch.zeros(self.target_shape)
        sigma1_target = -torch.zeros(self.target_shape)

        ## testing shape 
        self.check_shape(sigma_dot0, self.target_shape)
        self.check_shape(sigma_dot05, self.target_shape)
        self.check_shape(sigma_dot075, self.target_shape)
        self.check_shape(sigma_dot1, self.target_shape)
        
    ######################################################
    ############### PFFP-v1 Interpolant ##################
    ######################################################
    def test_surr_si_intrpl_pffp_v1_coeffs_alpha(self) -> None:
        ## computing alpha coefficient
        alpha0 = self.interpolant_v1.coeffs.alpha(self.t0)
        alpha05 = self.interpolant_v1.coeffs.alpha(self.t05)
        alpha075 = self.interpolant_v1.coeffs.alpha(self.t075)
        alpha1 = self.interpolant_v1.coeffs.alpha(self.t1)
    
        ## defining the target values
        alpha0_target = torch.ones(self.target_shape)
        alpha05_target = torch.zeros(self.target_shape) + 0.5
        alpha075_target = torch.zeros(self.target_shape) + 0.25
        alpha1_target = torch.zeros(self.target_shape)

        ## testing shape 
        self.check_shape(alpha0, self.target_shape)
        self.check_shape(alpha05, self.target_shape)
        self.check_shape(alpha075, self.target_shape)
        self.check_shape(alpha1, self.target_shape)
        
        ## testing target values
        self.check_equal(alpha0, alpha0_target)
        self.check_equal(alpha05, alpha05_target)
        self.check_equal(alpha075, alpha075_target)
        self.check_equal(alpha1, alpha1_target)

    def test_surr_si_intrpl_pffp_v1_coeffs_beta(self) -> None:
        ## computing alpha coefficient
        beta0 = self.interpolant_v1.coeffs.beta(self.t0)
        beta05 = self.interpolant_v1.coeffs.beta(self.t05)
        beta075 = self.interpolant_v1.coeffs.beta(self.t075)
        beta1 = self.interpolant_v1.coeffs.beta(self.t1)
    
        ## defining the target values
        beta0_target = self.t0
        beta05_target = self.t05
        beta075_target = self.t075
        beta1_target = self.t1

        ## testing shape 
        self.check_shape(beta0, self.target_shape)
        self.check_shape(beta05, self.target_shape)
        self.check_shape(beta075, self.target_shape)
        self.check_shape(beta1, self.target_shape)
        
        ## testing target values
        self.check_equal(beta0, beta0_target)
        self.check_equal(beta05, beta05_target)
        self.check_equal(beta075, beta075_target)
        self.check_equal(beta1, beta1_target)
    
    def test_surr_si_intrpl_pffp_v1_coeffs_sigma(self) -> None:
        ## computing alpha coefficient
        sigma0 = self.interpolant_v1.coeffs.sigma(self.t0)
        sigma05 = self.interpolant_v1.coeffs.sigma(self.t05)
        sigma075 = self.interpolant_v1.coeffs.sigma(self.t075)
        sigma1 = self.interpolant_v1.coeffs.sigma(self.t1)
    
        ## defining the target values
        sigma0_target = torch.ones(self.target_shape)
        sigma05_target = torch.zeros(self.target_shape) + 0.5
        sigma075_target = torch.zeros(self.target_shape) + 0.25
        sigma1_target = torch.zeros(self.target_shape)

        ## testing shape 
        self.check_shape(sigma0, self.target_shape)
        self.check_shape(sigma05, self.target_shape)
        self.check_shape(sigma075, self.target_shape)
        self.check_shape(sigma1, self.target_shape)
        
        ## testing target values
        self.check_equal(sigma0, sigma0_target)
        self.check_equal(sigma05, sigma05_target)
        self.check_equal(sigma075, sigma075_target)
        self.check_equal(sigma1, sigma1_target)
    
    ######################################################
    ################ PFFP-v1 Velocity ####################
    ######################################################
    def test_surr_si_intrpl_pffp_v1_coeffs_alpha_dot(self) -> None:
        ## computing alpha coefficient
        alpha_dot0 = self.interpolant_v1.coeffs.alpha_dot(self.t0)
        alpha_dot05 = self.interpolant_v1.coeffs.alpha_dot(self.t05)
        alpha_dot075 = self.interpolant_v1.coeffs.alpha_dot(self.t075)
        alpha_dot1 = self.interpolant_v1.coeffs.alpha_dot(self.t1)
    
        ## defining the target values
        alpha_dot0_target = -torch.ones(self.target_shape)
        alpha_dot05_target = -torch.zeros(self.target_shape)
        alpha_dot075_target = -torch.zeros(self.target_shape)
        alpha1_target = -torch.zeros(self.target_shape)

        ## testing shape 
        self.check_shape(alpha_dot0, self.target_shape)
        self.check_shape(alpha_dot05, self.target_shape)
        self.check_shape(alpha_dot075, self.target_shape)
        self.check_shape(alpha_dot1, self.target_shape)

    def test_surr_si_intrpl_pffp_v1_coeffs_beta_dot(self) -> None:
        ## computing alpha coefficient
        beta_dot0 = self.interpolant_v1.coeffs.alpha_dot(self.t0)
        beta_dot05 = self.interpolant_v1.coeffs.alpha_dot(self.t05)
        beta_dot075 = self.interpolant_v1.coeffs.alpha_dot(self.t075)
        beta_dot1 = self.interpolant_v1.coeffs.alpha_dot(self.t1)
    
        ## defining the target values
        beta_dot0_target = self.t0
        beta_dot05_target = self.t05
        beta_dot075_target = self.t075
        beta1_target = self.t1

        ## testing shape 
        self.check_shape(beta_dot0, self.target_shape)
        self.check_shape(beta_dot05, self.target_shape)
        self.check_shape(beta_dot075, self.target_shape)
        self.check_shape(beta_dot1, self.target_shape)
    
    def test_surr_si_intrpl_pffp_v1_coeffs_sigma_dot(self) -> None:
        ## computing alpha coefficient
        sigma_dot0 = self.interpolant_v1.coeffs.sigma_dot(self.t0)
        sigma_dot05 = self.interpolant_v1.coeffs.sigma_dot(self.t05)
        sigma_dot075 = self.interpolant_v1.coeffs.sigma_dot(self.t075)
        sigma_dot1 = self.interpolant_v1.coeffs.sigma_dot(self.t1)
    
        ## defining the target values
        sigma_dot0_target = -torch.ones(self.target_shape)
        sigma_dot05_target = -torch.zeros(self.target_shape)
        sigma_dot075_target = -torch.zeros(self.target_shape)
        sigma1_target = -torch.zeros(self.target_shape)

        ## testing shape 
        self.check_shape(sigma_dot0, self.target_shape)
        self.check_shape(sigma_dot05, self.target_shape)
        self.check_shape(sigma_dot075, self.target_shape)
        self.check_shape(sigma_dot1, self.target_shape)