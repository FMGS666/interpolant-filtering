"""
This files contains the common code shared between the module
It should not have any external dependencies within the package
"""
import torch
import random
import numpy as np

from pathlib import Path
from typing import Union

## defining configuration data type
ConfigDataTypes = Union[str, torch.Tensor, torch.nn.Module, float, int, list[int], bool]
ConfigData = dict[str, ConfigDataTypes]

## defining custom input/output data types
DictionaryData = dict[str, torch.Tensor]
InputData = Union[torch.Tensor, DictionaryData]
OutputData = Union[torch.Tensor, DictionaryData]

## defining custom model data type 
ModelData = Union[torch.nn.Module]

## defining custom path data type
PathData = Union[str, Path]