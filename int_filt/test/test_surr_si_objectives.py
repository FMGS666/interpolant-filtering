"""
File containing the test for the `surr_si.[interpolants, pffp_coefficients]`  module
"""
import torch

from .common import Test

from ..src import create_interpolant, create_models, DriftObjective
from ..utils import InputData


######################################################
################# Surr SI Objective ################## 
######################################################
class SurrSIObjectivesTest(Test):
    def setUp(self) -> None:
        ## defining spatial dims
        spatial_dims = 2
        ## initializing pffp-v0 interpolant
        si_config = {"method": "pffp_v0"}
        self.interpolant = create_interpolant(si_config)

        ## model configurations
        model_config = {
            "backbone": "mlp",
            "spatial_dims": spatial_dims,
            "b_net_hidden_dims": [16], 
            "b_net_activation": torch.nn.ReLU(),
            "b_net_activate_final": False,
        } 

        models = create_models(model_config)
        self.b_net = models["b_net"]

        ## defining mc configurations
        self.mc_config = {"num_samples": 100}

        ## initializing drift objective 
        Lb_config = {
            "model": self.b_net,
            "interpolant": self.interpolant,
            "mc_config": self.mc_config,
        }
        self.Lb = DriftObjective(Lb_config)

        ## defining expected size
        self.target_shape = torch.Size((16, 2))

    ######################################################
    ############## Surr-SI drift objective ###############
    ######################################################
    def test_surr_si_objctv_drift(self) -> None:
        ## simply trying a forward pass on an example batch
        x0 = torch.randn(self.target_shape)
        x1 = torch.randn(self.target_shape)*0.5 + 1.0
        y = x1 + 0.05*torch.randn(self.target_shape)
        batch = {"x0": x0, "x1": x1, "y": y}
        loss = self.Lb(batch)