"""
File containing common code for running experiment
"""
import torch

from typing import Optional
from tqdm import tqdm

from ..src import DriftObjective

from ..utils import ConfigData, InputData, OutputData, move_batch_to_device, construct_time_discretization

class Experiment:
    def __init__(self, config: ConfigData) -> None:
        """
        Constructor with custom config dictionary
        """
        ## initializing attributes
        self.config = config
        ## parsing configuration dictionary
        self.interpolant = self.config["interpolant"]
        self.b_net = self.config["b_net"]
        self.ssm = self.config["ssm"]
        self.writer = self.config["writer"]
        self.mc_config = self.config["mc_config"]
        self.device = self.config["device"]
        self.preprocessing = self.config["preprocessing"]
        self.logging_step = 5

    def get_batch(self) -> OutputData:
        """
        Samples a batch from the ssm
        """
        raise NotImplementedError

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
            "preprocessing": self.preprocessing
        }
        Lb = DriftObjective(Lb_config)
        ## defining iterator
        iterator = tqdm(range(num_grad_steps))
        ## starting optimization
        for grad_step in range(num_grad_steps):
            ## preparing batch
            batch = self.get_batch()
            batch = move_batch_to_device(batch, self.device)
            ## estimating loss
            loss = Lb.forward(batch)
            loss_value = loss.item()
            ## optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ## scheduler step 
            if scheduler is not None:
                scheduler.step()
            # retrieving learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            ## logging
            iterator.set_description(f"Grad Step {grad_step + 1}/{num_grad_steps}, MSELoss: {loss_value}, Learning Rate {current_lr}")
            iterator.update()
            if grad_step % self.logging_step == 0:
                self.writer.add_scalar("train/drift_loss", loss_value, grad_step)
                self.writer.add_scalar("train/learning_rate", current_lr, grad_step)
    
    def simulate_sde(self, batch: InputData, sample_config: ConfigData) -> OutputData:
        r"""
        Simulates the SDE $dX_t = b(t, X_t)dt + \sigma_tdB_t$
        """
        ## parsing configuration dictionary
        num_time_steps = sample_config["num_time_steps"]
        ## constructing time discretization
        time, stepsizes = construct_time_discretization(num_time_steps, self.device)
        ## augmenting batch
        batch["t"] = time[0]
        batch["xt"] = batch["x0"]
        ## preprocessing batch 
        batch = self.preprocessing(batch)
        ## computing drift
        drift = stepsizes[0]*self.b_net(batch)
        ## sampling noise 
        eta = torch.randn_like(drift)
        ## computing diffusion
        diffusion = self.interpolant.coeffs.sigma(time[0])*torch.sqrt(stepsizes[0])*eta
        # updating state and current batch
        x = batch["x0"]
        x = x + drift + diffusion
        # iterating over each step of the euler discretization
        for n in range(1, num_time_steps):
            # getting the stepsize
            delta_t = stepsizes[n]
            t = time[n]
            ## updating current batch
            batch["xt"] = x
            batch["t"] = t
            ## preprocessing batch 
            batch = self.preprocessing(batch)
            # computing adjusted drift
            drift = self.b_net(batch)
            drift = delta_t*drift
            # sampling noise
            eta = torch.randn_like(drift)
            # computing diffusion term
            diffusion = self.interpolant.coeffs.sigma(t)*torch.sqrt(delta_t)*eta
            # euler step
            x = x + drift + diffusion
        return x
    
    def sample(self, batch: InputData, sample_config: ConfigData) -> OutputData:
        r"""
        Samples  from the model by simulating the SDE $dX_t = b(t, X_t)dt + \sigma_tdB_t$
        """
        ## parsing configuration dictionary
        num_samples = sample_config["num_samples"]
        ## retrieving necessary data
        num_sims = self.ssm.num_sims
        num_dims = self.ssm.num_dims
        ## allocating memory
        samples_store = torch.zeros(num_samples, num_sims, num_dims)
        ## iterating over each sample
        for sample_id in tqdm(range(num_samples)):
            X1 = self.simulate_sde(batch, sample_config)
            samples_store[sample_id] = X1.detach().cpu()
        return samples_store

    def FA_APF(self, filter_conf: Optional[ConfigData] = None) -> OutputData:
        """
        Runs Fully Augmented Auxiliary Particle Filter
        """
        raise NotImplementedError