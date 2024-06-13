"""
This file contains utility functions needed across the code base
"""

import torch
import random
import numpy as np

## defining function time discretization
def construct_time_discretization(N, device, lbound = 0.0, ubound = 1.0):    
    time = torch.linspace(lbound, ubound, N + 1, device = device)
    stepsizes = ((ubound - lbound) / N) * torch.ones(N, device = device)
    return (time, stepsizes)

## defining function for reproducibility
def ensure_reproducibility(SEED):
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  
    torch.manual_seed(SEED)  
    np.random.seed(SEED)  
    random.seed(SEED)

## defining function for safe dim broadcast
## of an input tensor to a target tensor
## reshapes the input tensor to the shape of 
## the target one by repeating it in the missing dimensions. 
## for a tensor `target` of a given shape and an `input` tensor
## with `input.dim() < target.dim()` and such that
## `input.shape == target.shape[:input.dim()]`
## `unsqueeze_last_dim`: whether to simply unsqueeze the last missing dimension
def safe_broadcast(input, target, unsqueeze_last_dim = True, debug = False):
    ## retrieving shapes
    input_shape = input.shape
    target_shape = target.shape
    ## retrieving dim
    input_dim = input.dim()
    target_dim = target.dim()
    ## We also need to verify that the 
    ## input has a smaller number of dimensions than the target
    msg = f"The input dim should be not greater than the target dim (input {input_dim}, target {target_dim})"
    assert input_dim <= target_dim, msg
    ## verifying that they have the same shape
    ## in the shared dimensions
    msg = lambda d: f"The input and target differ in the minibatch size (input: {input_shape[d]}, target: {target_shape[d]})"
    for d in range(input_dim):
        assert input_shape[d] == target_shape[d], msg(d)
    ## definig the number of dimension to broadcast after sanity checks
    num_dims_to_broadcast = target_dim - input_dim
    ## early return if no broadcast needed
    if num_dims_to_broadcast == 0:
        return input
    ## reshaping target according to target shape
    ## iterating over the dimensions to broadcast
    for dim_to_broadcast in range(input_dim, input_dim + num_dims_to_broadcast):
        ## displaying debugging message
        if debug:
            print(f"safe_dim_broadcast::{dim_to_broadcast}->", input.shape)
        ## adding new dimension
        input = torch.unsqueeze(input, dim = -1)
        ## skipping last dimension after having 
        ## unsqueezed if this is what we want
        if unsqueeze_last_dim and (dim_to_broadcast == input_dim + num_dims_to_broadcast - 1):
            continue
        ## defining number of repetition per 
        ## additional dimension and broadcasting input
        ## base will only repeat tensor once (not repeating)
        ## for the dimensions already matching the shape
        base_broadcast = [1 for _ in range(dim_to_broadcast)]
        ## retrieving the number of repetitions for 
        ## current extra dimension to broadcast
        current_target = target_shape[dim_to_broadcast]
        current_broadcast = base_broadcast + [current_target]
        input = input.repeat(current_broadcast)
    return input

## defining function for safe batch concatenation
def safe_cat(batch, cat_keys, to_broadcast = [], dim = -1, unsqueeze_last_dim = True, debug = False):
    ## defining list of tensor to concatenate
    to_cat = []
    ## retrieving reference keys for computing target shape 
    reference_keys = [k for k  in cat_keys if k not in to_broadcast]
    reference_tensor = batch[reference_keys[0]]
    reference_shape = reference_tensor.shape
    for k in reference_keys:
        tensor = batch[k]
        current_shape = tensor.shape
        ## defining assertion message
        msg = lambda curr_shape, ref_shape: f"Current shape {curr_shape} differs from {ref_shape}"
        ## sanity check: the keys to concatenate need to have the 
        ## same shape if they do not appear in `to_broadcast`
        assert np.all(current_shape == reference_shape), msg(current_shape, reference_shape)
        to_cat.append(tensor)
        ## displaying debugging message
        if debug:
            print(f"safe_cat::{k}->", tensor.shape)
    ## now need to broadcast required keys to 
    ## match the target shape
    for k in to_broadcast:
        tensor = batch[k]
        tensor = safe_broadcast(tensor, reference_tensor, unsqueeze_last_dim = unsqueeze_last_dim)
        #tensor = torch.unsqueeze(tensor, dim = dim)
        ## defining assertion message
        msg = lambda curr_dim, ref_dim: f"Current dim {curr_dim} differs from {ref_dim}"
        assert tensor.dim() == reference_tensor.dim(), msg(tensor.dim(), reference_tensor.dim())
        to_cat.append(tensor)
        ## displaying debugging message
        if debug:
            print(f"safe_cat::{k}->", tensor.shape)
    ## displaying debugging message
    if debug:
        for tensor in to_cat:
            print(f"OUTPUT.safe_cat::", tensor.shape)
    ## now we can perform concatenation 
    xcat = torch.cat(to_cat, dim = dim)
    return xcat

## function for moving batch of tensors to device
def move_batch_to_device(batch, device):
    batch_copy = dict()
    for key, tensor in batch.items():
        tensor = tensor.to(device)
        batch_copy[key] = tensor
    return batch_copy