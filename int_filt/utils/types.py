"""
This files contains the common code shared between the module
It should not have any external dependencies within the package
"""
import torch
import random
import numpy as np

from typing import Union

## defining configuration data type
ConfigDataTypes = Union[str, torch.Tensor, float, int, list[int], bool]
ConfigData = dict[str, ConfigDataTypes]

## defining custom input/output data types
DictionaryData = dict[str, torch.Tensor]
InputData = Union[torch.Tensor, DictionaryData]
OutputData = Union[torch.Tensor, DictionaryData]

## defining custom model data type 
ModelData = Union[torch.nn.Module]