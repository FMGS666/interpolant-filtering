"""
Script for running the experiment on the Ornstein-Uhlenbeck model
"""
import torch

from .common import Experiment

from ..src import DriftObjective

from ..utils import ConfigData, OutputData, move_batch_to_device

class OUExperiment(Experiment):
    """
    Class for handling the experiments on the Ornstein-Uhlenbeck model
    """
    def __init__(self, config: ConfigData):
        """
        Constructor with custom config dictionary
        """
        super(OUExperiment, self).__init__(config)
        ## parsing configuration dictionary
        self.interpolant = self.config["interpolant"]
        self.b_net = self.config["b_net"]
        self.ssm = self.config["ssm"]
        self.writer = self.config["writer"]
        self.mc_config = self.config["mc_config"]
        self.device = self.config["device"]
        self.logging_step = 5
    
    def train(self, optim_config: ConfigData) -> OutputData:
        """
        Trains the $b$ model
        """
        ## parsing configuration dictionary
        num_grad_steps = optim_config["num_grad_steps"]
        optimizer = optim_config["optimizer"]
        scheduler = optim_config["scheduler"]
        ## initializing objective function
        Lb_config = {
            "b_net": self.b_net, 
            "interpolant": self.interpolant, 
            "mc_config": self.mc_config
        }
        Lb = DriftObjective(Lb_config)
        ## defining store loss
        loss_history = torch.zeros((num_grad_steps))
        ## starting optimization
        for grad_step in range(num_grad_steps):
            ## sampling from stationary state distribution
            x0 = self.ssm.initial_state().float()
            x1 = self.ssm.initial_state().float()
            xc = x0.float()
            ## sampling from observation model
            y = self.ssm.observation().float()
            ## constructing batch
            batch = {"x0": x0, "x1": x1, "xc": xc, "y": y}
            batch = move_batch_to_device(batch, self.device)
            ## estimating loss
            loss = Lb.forward(batch)
            loss_value = loss.item()
            ## optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ## storing loss
            loss_history[grad_step] = loss_value
            ## logging
            if grad_step % self.logging_step == 0:
                self.writer.add_scalar("train/drift_loss", loss_value, grad_step)
                print(f"Grad step: {grad_step}, Velocity Field MSE Loss: {loss_value}", flush = True)
        return loss_history