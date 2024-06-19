"""
File containing common code for running experiment
"""
import torch
import os
import gc

from typing import Optional
from tqdm import tqdm

from ..src import DriftObjective

from ..utils import ConfigData, InputData, OutputData, move_batch_to_device, construct_time_discretization, clone_batch

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
        self.log_results = self.config["log_results"]
        self.logging_step = self.config["logging_step"]
        self.full_out = self.config["full_out"]
        self.clear_memory = self.config["clear_memory"]

    def get_batch(self, train: Optional[bool] = None, idx: Optional[int] = None) -> OutputData:
        """
        Samples a batch from the ssm
        """
        raise NotImplementedError

    def train(self, config: ConfigData) -> OutputData:
        """
        Trains the $b$ model
        """
        ## retrieving optimizer and scheduler
        optimizer = config["optimizer"]
        scheduler = config["scheduler"]
        ## initializing objective function
        Lb_config = {
            "b_net": self.b_net, 
            "interpolant": self.interpolant, 
            "mc_config": self.mc_config,
            "preprocessing": self.preprocessing,
            "full_out": self.full_out
        }
        Lb = DriftObjective(Lb_config)
        ## allocating memory for storing loss and lr
        loss_history = torch.zeros((config["num_grad_steps"]))
        lr_history = torch.zeros((config["num_grad_steps"]))
        if self.full_out:
            drift_store_history = torch.zeros((config["num_grad_steps"], self.mc_config["num_mc_samples"], self.ssm.num_sims, self.ssm.num_dims))
        ## defining iterator
        iterator = tqdm(range(config["num_grad_steps"]))
        ## starting optimization
        for grad_step in range(config["num_grad_steps"]):
            ## preparing batch
            batch = self.get_batch()
            batch = move_batch_to_device(batch, self.device)
            ## estimating loss
            loss_dict = Lb.forward(batch)
            # parsing loss dictionary
            loss = loss_dict["loss"]
            if self.full_out:
                drift_store = loss_dict["drift_store"]
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
            if self.full_out:
                drift_store_history[grad_step] = drift_store
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
        if self.full_out:
            train_dict["drift_store_history"] = drift_store_history
        return train_dict

    def simulate_sde(self, batch: InputData, config: ConfigData) -> OutputData:
        r"""
        Simulates the SDE $dX_t = b(t, X_t)dt + \sigma_tdB_t$
        """
        ## retrieving necessary data
        num_sims = self.ssm.num_sims
        num_dims = self.ssm.num_dims
        ## constructing time discretization
        time, stepsizes = construct_time_discretization(config["num_time_steps"], self.device)
        ## allocating memory
        if self.full_out:
            trajectory = torch.zeros((config["num_time_steps"], num_sims, num_dims))
            drift_history = torch.zeros((config["num_time_steps"], num_sims, num_dims))
            diffusion_history = torch.zeros((config["num_time_steps"], num_sims, num_dims))
        ## retrieving the starting state
        x = batch["x0"].clone()
        # iterating over each step of the euler discretization
        for n in range(config["num_time_steps"]):
            ## cloning the batch
            batch_clone = clone_batch(batch)
            # getting the time and stepsize
            delta_t = stepsizes[n]
            t = time[n]
            ## updating current batch
            batch_clone["xt"] = x
            batch_clone["t"] = t
            ## preprocessing batch 
            batch_clone = self.preprocessing(batch_clone)
            # computing adjusted drift
            with torch.no_grad():
                drift = self.b_net(batch_clone)
            # sampling noise
            eta = torch.randn_like(drift)
            # computing diffusion term
            diffusion = self.interpolant.coeffs.sigma(t)*torch.sqrt(delta_t)*eta   
            # euler step
            x = x + delta_t*drift + diffusion   
            ## storing state
            if self.full_out:
                trajectory[n] = x
                drift_history[n] = drift
                diffusion_history[n] = diffusion
        ## constructing output dictionary
        sde_dict = {"x": x}
        if self.full_out:
            sde_dict["trajectory"] = trajectory
            sde_dict["drift"] = drift
            sde_dict["diffusion"] = diffusion
        return sde_dict
    
    def sample(self, batch: InputData, config: ConfigData) -> OutputData:
        r"""
        Samples  from the model by simulating the SDE $dX_t = b(t, X_t)dt + \sigma_tdB_t$
        """
        ## retrieving necessary data
        num_sims = self.ssm.num_sims
        num_dims = self.ssm.num_dims
        ## allocating memory
        samples_store = torch.zeros(config["num_samples"], num_sims, num_dims)
        if self.full_out:
            trajectory_store = torch.zeros(config["num_samples"], config["num_time_steps"], num_sims, num_dims)
            drift_store = torch.zeros(config["num_samples"], config["num_time_steps"], num_sims, num_dims)
            diffusion_store = torch.zeros(config["num_samples"], config["num_time_steps"], num_sims, num_dims)
        ## defining iterator
        iterator = tqdm(range(config["num_samples"]))
        ## iterating over each sample
        for sample_id in iterator:
            ## simulating sde
            sde_dict = self.simulate_sde(batch, config)
            ## storing results
            samples_store[sample_id] = sde_dict["x"].clone().detach().cpu()
            if self.full_out:
                trajectory_store[sample_id] = sde_dict["trajectory"].clone().detach().cpu()
                drift_store[sample_id] = sde_dict["drift"].clone().detach().cpu()
                diffusion_store[sample_id] = sde_dict["diffusion"].clone().detach().cpu()
            ## cleaning up memory
            if self.clear_memory:
                del sde_dict 
                gc.collect()
                torch.cuda.empty_cache()
        ## constructing output dictionary
        sample_dict = {"samples": samples_store}
        if self.full_out:
            sample_dict["trajectory"] = trajectory_store
            sample_dict["drift"] =  drift_store
            sample_dict["diffusion"] = diffusion_store 
        return sample_dict
    
    def ar_sample(self, batch: Optional[InputData] = None, config: Optional[ConfigData] = None) -> OutputData:
        """
        Performs autoregressive sampling
        """
        ## getting the test initial state
        if batch is None:
            batch = experiment.get_batch(train = config["ar_sample_train"], idx = config["initial_time_step"])
            batch = move_batch_to_device(batch, experiment.device)
        ## retrieving necessary data
        num_sims = self.ssm.num_sims
        num_dims = self.ssm.num_dims
        ## cloning the batch
        batch_clone = clone_batch(batch)
        ## allocating memory
        ar_samples_store = torch.zeros((config["num_ar_steps"], num_sims, num_dims))
        if self.full_out:
            trajectory_store = torch.zeros((config["num_ar_steps"], config["num_time_steps"], num_sims, num_dims))
            drift_store = torch.zeros((config["num_ar_steps"], config["num_time_steps"], num_sims, num_dims))
            diffusion_store = torch.zeros((config["num_ar_steps"], config["num_time_steps"], num_sims, num_dims))
        ## defining iterator
        iterator = tqdm(range(config["num_ar_steps"]))
        ## iterating over each ar step
        for ar_step in iterator:
            ## cloning the batch
            batch_clone = clone_batch(batch)
            ## constructing sampling configuration dictionary
            sde_config = {"num_time_steps": config["num_time_steps"]}
            ## simulating sde
            sde_dict = self.simulate_sde(batch_clone, sde_config)
            ## updating batch
            x = sde_dict["x"].clone().detach()
            batch_clone["x0"] = x
            batch_clone["xc"] = x
            ## storing results
            ar_samples_store[ar_step] = x
            if self.full_out:
                trajectory_store[ar_step] = sde_dict["trajectory"].clone().detach().cpu()
                drift_store[ar_step] = sde_dict["drift"].clone().detach().cpu()
                diffusion_store[ar_step] = sde_dict["diffusion"].clone().detach().cpu()
            ## cleaning up memory
            if self.clear_memory:
                del sde_dict 
                gc.collect()
                torch.cuda.empty_cache()
        ## constructing output dictionary
        sample_dict = {"ar_samples": ar_samples_store}
        if self.full_out:
            sample_dict["trajectory"] =  trajectory_store
            sample_dict["drift"] =  drift_store
            sample_dict["diffusion"] = diffusion_store 
        return sample_dict

    def FA_APF(self, filter_conf: Optional[ConfigData] = None) -> OutputData:
        """
        Runs Fully Augmented Auxiliary Particle Filter
        """
        raise NotImplementedError