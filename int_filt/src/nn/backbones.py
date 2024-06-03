"""
This file contains the implementation of the backbone models used in the experiments
"""
import torch

from ...utils import ConfigData, ModelData, InputData, OutputData

class MLP(torch.nn.Module):
    """
    Class implementing the MLP
    """
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary, required keys:
            * `input_dim`, 
            * `hidden_dims`,
            * `out_dims`,
            * `activation`,
            * `activate_final`
        """
        super(MLP, self).__init__()
        ## initializing attributes
        self.config = config
        ## parsing configuration dictionary
        self.input_dim = self.config["input_dim"]
        self.hidden_dims = self.config["hidden_dims"]
        self.output_dim = self.config["output_dim"]
        self.activation = self.config["activation"]
        self.activate_final = self.config["activate_final"]
        ## defining optional final activation to identity
        self.final_activation = torch.nn.Identity()
        if self.activate_final:
            self.final_activation = self.activation
        ## initializing layers
        self.layers = self.init_layers()
    
    def init_layers(self) -> ModelData:
        """
        Initializes the model's layers
        """
        ## initializing layers
        layer_widths = [self.input_dim] + self.hidden_dims + [self.output_dim]
        layers = torch.nn.ModuleList()
        for layer_idx, input_width in enumerate(layer_widths[:-1]):
            output_width = layer_widths[layer_idx + 1]
            layer = torch.nn.Linear(input_width, output_width)
            layers.append(layer)
        return layers

    def forward(self, x: InputData) -> OutputData:
        """
        Performs forward pass on a batch of data
        """
        ## iterate over layer to perfrom forward pass
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        ## optional final activation
        x = self.final_activation(x)
        return x