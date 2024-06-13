"""
Script for running the experiment on the Ornstein-Uhlenbeck model
"""
import torch

from .common import Experiment

from ..src import DriftObjective

from ..utils import ConfigData, InputData, OutputData, move_batch_to_device

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
    
    def get_batch(self) -> OutputData:
        """
        Samples a batch from the ssm
        """
        ## sampling from stationary state distribution
        x0 = self.ssm.initial_state().float()
        x1 = self.ssm.initial_state().float()
        xc = x0.float()
        ## sampling from observation model
        y = self.ssm.observation().float()
        ## constructing batch
        batch = {"x0": x0, "x1": x1, "xc": xc, "y": y}
        return batch

    def standardize(self, batch: InputData) -> OutputData:
        """
        standardizes a batch of data
        """
        ## defining keys for latent states
        latent_state_keys = ["x0", "x1", "xc"]
        observation_keys = ["y"]
        skip_keys = ["t"]
        ## retrieving necessary data
        sigma_x = self.ssm.sigma_x
        sigma_y = self.ssm.sigma_y
        beta = self.ssm.beta
        ## normalizing batch
        batch_copy = dict()
        for key, tensor in batch.items():
            ## computing target std
            if key in latent_state_keys:
                std = (sigma_x / np.sqrt(2.0*beta)) * torch.ones_like(tensor)
            elif key in observation_keys:
                std = np.sqrt(sigma_x**2 / (2.0 * beta) + sigma_y**2) * torch.ones_like(tensor)
            elif key in skip_keys:
                continue
            ## scaling tensor
            tensor = tensor / std
            batch_copy[key] = tensor
        return batch_copy

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
            "mc_config": self.mc_config,
            "preprocess_batch": self.standardize
        }
        Lb = DriftObjective(Lb_config)
        ## defining store loss
        loss_history = torch.zeros((num_grad_steps))
        ## defining iterator
        iterator = tqdm(range(num_grad_steps))
        ## starting optimization
        for grad_step in range(num_grad_steps):
            ## preparing batch
            batch = self.get_batch()
            #batch = self.standardize(batch)
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
            iterator.set_description(f"Grad Step {grad_step + 1}/{num_grad_steps}, MSELoss: {loss_value}")
            iterator.update()
            if grad_step % self.logging_step == 0:
                self.writer.add_scalar("train/drift_loss", loss_value, grad_step)
        return loss_history