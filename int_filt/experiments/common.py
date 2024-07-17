"""
File containing common code for running experiment
"""
import torch
import os
import gc

from typing import Optional
from tqdm import tqdm

from ..src import DriftObjective, ControlledDriftObjective

from ..utils import (
    ConfigData,
    InputData,
    OutputData, 
    move_batch_to_device, 
    construct_time_discretization, 
    clone_batch, 
    safe_broadcast,
    resampling
)

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
        self.postprocessing = self.config["postprocessing"]
        self.log_results = self.config["log_results"]
        self.logging_step = self.config["logging_step"]
        self.full_out = self.config["full_out"]
        self.clear_memory = self.config["clear_memory"]
        ## optional attributes
        self.c_net = None
        if "c_net" in self.config.keys():
            self.c_net = self.config["c_net"]
        self.observation_model = None
        if "observation_model" in self.config.keys():
            self.observation_model = self.config["observation_model"]

    def get_batch(self, train: Optional[bool] = None, idx: Optional[int] = None) -> OutputData:
        """
        Samples a batch from the ssm
        """
        raise NotImplementedError

    def train_drift(self, config: ConfigData) -> OutputData:
        """
        Trains the $b$ model
        """
        ## retrieving optimizer and scheduler
        optimizer = config["b_net_optimizer"]
        scheduler = config["b_net_scheduler"]
        ## initializing objective function
        objective_config = {
            "b_net": self.b_net, 
            "interpolant": self.interpolant, 
            "mc_config": self.mc_config,
            "preprocessing": self.preprocessing,
        }
        objective = DriftObjective(objective_config)
        ## allocating memory for storing loss and lr
        loss_history = torch.zeros((config["num_grad_steps"]))
        lr_history = torch.zeros((config["num_grad_steps"]))
        ## defining iterator
        iterator = tqdm(range(config["num_grad_steps"]))
        ## starting optimization
        for grad_step in range(config["num_grad_steps"]):
            ## preparing batch
            batch = self.get_batch()
            batch = move_batch_to_device(batch, self.device)
            ## estimating loss
            loss_dict = objective.forward(batch)
            # parsing loss dictionary
            loss = loss_dict["loss"]
            ## retrieving loss value
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
            ## progress bar
            msg = f"Grad Step {grad_step + 1}/{config["num_grad_steps"]}, MSELoss: {loss_value}, Learning Rate {current_lr}"
            iterator.set_description(msg)
            iterator.update()
            ## storing loss and lr and sampled drifts
            loss_history[grad_step] = loss_value
            lr_history[grad_step] = current_lr
            ## logging
            if self.log_results and (grad_step % self.logging_step == 0):
                self.writer.add_scalar("train/drift_loss", loss_value, grad_step)
                self.writer.add_scalar("train/learning_rate", current_lr, grad_step)
            ## cleaning up memory
            if self.clear_memory:
                del batch, loss_dict 
                gc.collect()
                torch.cuda.empty_cache()
        ## constructing output dictionary
        train_dict = {"loss_history": loss_history, "lr_history": lr_history}
        return train_dict

    def train_controlled(self, config: ConfigData) -> OutputData:
        """
        Trains the $b$ model
        """
        ## retrieving optimizer and scheduler
        b_net_optimizer = config["b_net_optimizer"]
        b_net_scheduler = config["b_net_scheduler"]
        c_net_optimizer = config["c_net_optimizer"]
        c_net_scheduler = config["c_net_scheduler"]
        ## initializing objective function
        objective_config = {
            "b_net": self.b_net,
            "c_net": self.c_net, 
            "interpolant": self.interpolant, 
            "mc_config": self.mc_config,
            "preprocessing": self.preprocessing,
        }
        objective = ControlledDriftObjective(objective_config)
        ## allocating memory for storing loss and lr
        loss_history = torch.zeros((config["num_grad_steps"]))
        b_loss_history = torch.zeros((config["num_grad_steps"]))
        c_loss_history = torch.zeros((config["num_grad_steps"]))
        b_lr_history = torch.zeros((config["num_grad_steps"]))
        c_lr_history = torch.zeros((config["num_grad_steps"]))
        ## defining iterator
        iterator = tqdm(range(config["num_grad_steps"]))
        ## starting optimization
        for grad_step in range(config["num_grad_steps"]):
            ## preparing batch
            batch = self.get_batch()
            batch = move_batch_to_device(batch, self.device)
            ## estimating loss
            loss_dict = objective.forward(batch)
            # parsing loss dictionary
            loss = loss_dict["loss"]
            b_loss = loss_dict["b_loss"]
            c_loss = loss_dict["c_loss"]
            if self.full_out:
                drift_store = loss_dict["drift_store"]
            ## retrieving loss value
            loss_value = loss.item()
            b_loss_value = b_loss.item()
            c_loss_value = c_loss.item()
            ## optimization step
            b_net_optimizer.zero_grad()
            c_net_optimizer.zero_grad()
            loss.backward()
            b_net_optimizer.step()
            c_net_optimizer.step()
            ## scheduler step 
            if b_net_scheduler is not None:
                b_net_scheduler.step()
            if c_net_scheduler is not None:
                c_net_scheduler.step()
            # retrieving learning rate
            b_current_lr = b_net_optimizer.param_groups[0]["lr"]
            c_current_lr = c_net_optimizer.param_groups[0]["lr"]
            ## progress bar
            msg = f"Grad Step {grad_step + 1}/{config["num_grad_steps"]}, MSELoss: {loss_value}, Drift Loss: {b_loss_value}, Control Loss: {c_loss_value}, BNet lr: {b_current_lr}, CNet lr: {c_current_lr}"
            iterator.set_description(msg)
            iterator.update()
            ## storing loss and lr and sampled drifts
            loss_history[grad_step] = loss_value
            b_loss_history[grad_step] = b_loss_value
            c_loss_history[grad_step] = c_loss_value
            b_lr_history[grad_step] = b_current_lr
            c_lr_history[grad_step] = c_current_lr
            ## logging
            if self.log_results and (grad_step % self.logging_step == 0):
                self.writer.add_scalar("train/loss", loss_value, grad_step)
                self.writer.add_scalar("train/drift_loss", b_loss_value, grad_step)
                self.writer.add_scalar("train/control_loss", c_loss_value, grad_step)
                self.writer.add_scalar("train/b_learning_rate", b_current_lr, grad_step)
                self.writer.add_scalar("train/c_learning_rate", c_current_lr, grad_step)
            ## cleaning up memory
            if self.clear_memory:
                del batch, loss_dict 
                gc.collect()
                torch.cuda.empty_cache()
        ## constructing output dictionary
        train_dict = {"loss_history": loss_history, "b_loss_history": b_loss_history, "c_loss_history": c_loss_history, "b_lr_history": b_lr_history, "c_lr_history": c_lr_history}
        return train_dict
  
    def simulate_sde(self, batch: InputData, config: ConfigData) -> OutputData:
        r"""
        Simulates the SDE $dX_t = b(t, X_t)dt + \sigma_tdB_t$
        """
        ## parsing batch dictionary
        x = batch["x0"]
        xc = batch["xc"]
        y = batch["y"]
        ## retrieving necessary data
        num_sims = x.shape[0]
        num_dims = self.ssm.num_dims
        ## constructing time discretization
        time, stepsizes = construct_time_discretization(config["num_time_steps"], self.device)
        ## allocating memory
        if self.full_out:
            trajectory = torch.zeros((config["num_time_steps"], num_sims, num_dims))
        # iterating over each step of the euler discretization
        for n in range(config["num_time_steps"]):
            ## cloning data for current iteration
            x_clone = x.clone()
            xc_clone = xc.clone()
            y_clone = y.clone()
            # getting the current time and stepsize
            delta_t = stepsizes[n]
            t = time[n]
            # sampling noise
            eta = torch.randn_like(x)
            ## computing weiner perturbation
            W = torch.sqrt(delta_t)*eta   
            ## computing interpolant coefficients
            sigma_t = self.interpolant.coeffs.sigma(t)
            sigma_t = safe_broadcast(sigma_t, x)
            ## constructing current batch
            current_batch = {"t": t, "xc": xc_clone, "xt": x_clone, "y": y_clone}
            ## preprocessing batch 
            current_batch = self.preprocessing.standardize(current_batch)
            ## forward pass on the model
            with torch.no_grad():
                drift = self.b_net(current_batch)
            ## postprocessing the batch
            if self.postprocessing:
                drift = self.preprocessing.unstandardize(drift)
            # computing diffusion term
            diffusion = sigma_t*W  
            # euler step
            x = x + delta_t*drift + diffusion   
            ## storing state
            if self.full_out:
                trajectory[n] = x.detach().cpu()
        ## cleaning up memory
        if self.clear_memory:
            del batch, current_batch, drift, diffusion 
            gc.collect()
            torch.cuda.empty_cache()
        ## constructing output dictionary
        sde_dict = {"x": x.detach()}
        if self.full_out:
            sde_dict["trajectory"] = trajectory
        return sde_dict
    
    def simulate_controlled_sde(self, batch: InputData, config: ConfigData) -> OutputData:
        r"""
        Simulates the SDE $dX_t = b(t, X_t)dt + \sigma_tdB_t$
        """
        ## parsing batch dictionary
        x = batch["x0"]
        xc = batch["xc"]
        y = batch["y"]
        ## retrieving necessary data
        num_sims = x.shape[0]
        num_dims = self.ssm.num_dims
        ## constructing time discretization
        time, stepsizes = construct_time_discretization(config["num_time_steps"], self.device)
        ## allocating memory
        if self.full_out:
            trajectory = torch.zeros((config["num_time_steps"], num_sims, num_dims))
            value_trajectory = torch.zeros((config["num_time_steps"], num_sims, 1))
        ## defining initial value process
        v = torch.zeros((num_sims, 1), device = self.device)
        # iterating over each step of the euler discretization
        for n in range(config["num_time_steps"]):
            ## cloning data for current iteration
            x_clone = x.clone()
            xc_clone = xc.clone()
            y_clone = y.clone()
            # getting the current time and stepsize
            delta_t = stepsizes[n]
            t = time[n]
            # sampling noise
            eta = torch.randn_like(x)
            ## computing weiner perturbation
            W = torch.sqrt(delta_t)*eta  
            ## computing interpolant coefficients
            sigma_t = self.interpolant.coeffs.sigma(t)
            sigma_t = safe_broadcast(sigma_t, x)
            ## constructing current batch
            current_batch = {"t": t, "xc": xc_clone, "xt": x_clone, "y": y_clone}
            ## preprocessing batch 
            current_batch = self.preprocessing.standardize(current_batch)
            ## forward pass on the models
            with torch.no_grad():
                drift = self.b_net(current_batch)
            Z = self.c_net(current_batch)
            ## postprocessing the batch
            if self.postprocessing:
                drift = self.preprocessing.unstandardize(drift)
                Z = self.preprocessing.unstandardize(Z)
            control = -Z.clone().detach()
            ## computing controlled drift and diffusion
            controlled_drift = drift + sigma_t*control
            diffusion = sigma_t*W
            ## computing value drift and diffusion coefficients
            value_drift = -0.5 * torch.sum(torch.square(Z), 1)
            value_drift = torch.unsqueeze(value_drift, dim = 1)
            value_diffusion = torch.sum(Z * W, 1)
            value_diffusion = torch.unsqueeze(value_diffusion, dim = 1)
            ## euler step
            x = x + delta_t*controlled_drift + diffusion
            v = v + delta_t*value_drift + value_diffusion
            ## storing state
            if self.full_out:
                trajectory[n] = x.detach().cpu()
                value_trajectory[n] = v.detach().cpu()
        ## cleaning up memory
        if self.clear_memory:
            del batch, current_batch, drift, diffusion, value_drift, value_diffusion,  
            gc.collect()
            torch.cuda.empty_cache()
        ## constructing output dictionary
        sde_dict = {"x": x.detach(), "v": v.detach()}
        if self.full_out:
            sde_dict["trajectory"] = trajectory
            sde_dict["value_trajectory"] = value_trajectory
        return sde_dict
  
    def sample(self, batch: InputData, config: ConfigData) -> OutputData:
        r"""
        Samples  from the model by simulating the SDE $dX_t = b(t, X_t)dt + \sigma_tdB_t$
        """
        ## retrieving necessary data
        num_sims = batch["x0"].shape[0]
        num_dims = self.ssm.num_dims
        ## allocating memory
        samples_store = torch.zeros(config["num_samples"], num_sims, num_dims)
        if self.full_out:
            trajectory_store = torch.zeros(config["num_samples"], config["num_time_steps"], num_sims, num_dims)
        ## defining iterator
        iterator = tqdm(range(config["num_samples"]))
        ## iterating over each sample
        for sample_id in iterator:
            ## simulating sde
            sde_dict = self.simulate_sde(batch, config)
            ## storing results
            samples_store[sample_id] = sde_dict["x"].cpu()
            if self.full_out:
                trajectory_store[sample_id] = sde_dict["trajectory"]
            ## cleaning up memory
            if self.clear_memory:
                del sde_dict 
                gc.collect()
                torch.cuda.empty_cache()
        ## constructing output dictionary
        sample_dict = {"samples": samples_store}
        if self.full_out:
            sample_dict["trajectory"] = trajectory_store
        return sample_dict
    
    def ar_sample(self, batch: Optional[InputData] = None, config: Optional[ConfigData] = None) -> OutputData:
        """
        Performs autoregressive sampling
        """
        ## parsing batch dictionary
        xc = batch["xc"]
        x = batch["x0"]
        y = batch["y"]
        ## retrieving necessary data
        num_sims = x.shape[0]
        num_dims = self.ssm.num_dims
        ## constructing sde configuration dictionary
        sde_config = {"num_time_steps": config["num_time_steps"]}
        ## allocating memory
        ar_samples_store = torch.zeros((config["num_ar_steps"], num_sims, num_dims))
        if self.full_out:
            trajectory_store = torch.zeros((config["num_ar_steps"], config["num_time_steps"], num_sims, num_dims))
        ## defining iterator
        iterator = tqdm(range(config["num_ar_steps"]))
        ## iterating over each ar step
        for ar_step in iterator:
            ## constructing current batch
            current_batch = {"x0": x, "xc": x, "y": y}
            ## simulating sde
            sde_dict = self.simulate_sde(current_batch, sde_config)
            ## parsing sde dictionary        
            x = sde_dict["x"]
            ## storing results
            ar_samples_store[ar_step] = x.cpu()
            if self.full_out:
                trajectory_store[ar_step] = sde_dict["trajectory"]
            ## cleaning up memory
            if self.clear_memory:
                del sde_dict 
                gc.collect()
                torch.cuda.empty_cache()
        ## constructing output dictionary
        sample_dict = {"ar_samples": ar_samples_store}
        if self.full_out:
            sample_dict["trajectory"] =  trajectory_store
        return sample_dict

    def FA_APF(self, batch: InputData, config: Optional[ConfigData]) -> OutputData:
        """
        Runs Fully Augmented Auxiliary Particle Filter
        """
        ## parsing batch dictionary
        xc = batch["xc"]
        x = batch["x0"]
        y_store = batch["y"]
        ## constructing sde configuration dictionary
        sde_config = {"num_time_steps": config["num_time_steps"]}
        ## allocating memory
        ess_store = torch.zeros((config["num_obs"] + 1))
        log_norm_const = torch.zeros((config["num_obs"] + 1))
        states = torch.zeros((config["num_particles"], config["num_obs"] + 1, self.ssm.num_dims))
        ## preparing for filtering
        log_ratio_norm_const = torch.tensor([0.0], device = self.device)
        ess_store[0] = config["num_particles"]
        states[:, 0, :] = batch["x0"]
        ## defining iterator over ssm time steps
        iterator = tqdm(range(config["num_obs"]))
        ## iterating over each observation step
        for observation_idx in iterator:
            ## retrieving current observation
            y = y_store[observation_idx]
            y = y.repeat((config["num_particles"], 1))
            ## constructing current batch
            current_batch = {"x0": x, "xc": x, "y": y}
            ## simulating controlled sde
            controlled_sde_dict = self.simulate_controlled_sde(current_batch, sde_config)
            ## parsing sde dictionary
            x = controlled_sde_dict["x"]
            v = controlled_sde_dict["v"]
            ## computing likelihood
            likelihood_batch = {"x": x, "y": y}
            likelihood = self.observation_model.log_prob(likelihood_batch)
            likelihood = torch.unsqueeze(likelihood, dim = 1)
            ## computing normalized weights
            log_weights = v + likelihood
            max_log_weights = torch.max(log_weights)
            weights = torch.exp(log_weights - max_log_weights)
            normalized_weights = weights / torch.sum(weights)
            ## computing ess and elbo
            ess = 1.0 / torch.sum(normalized_weights**2)
            log_ratio_norm_const = log_ratio_norm_const + torch.log(torch.mean(weights)) + max_log_weights
            ## resampling
            normalized_weights = normalized_weights.detach().cpu()
            ancestors = resampling(normalized_weights, config["num_particles"])
            x = x[ancestors,:]
            ## storing iteration data
            log_norm_const[observation_idx + 1] = log_ratio_norm_const
            ess_store[observation_idx + 1] = ess
            states[:, observation_idx + 1] = x
            ## cleaning up memory
            if self.clear_memory:
                del (
                    controlled_sde_dict, likelihood, log_weights, weights, normalized_weights,
                    ess, v
                ) 
                gc.collect()
                torch.cuda.empty_cache()
        ## contructing output dictionary
        out_dict = {"states": states, "ess": ess_store, "log_norm_const": log_norm_const}
        return out_dict