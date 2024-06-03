"""
File containing common code needed for test
"""
import torch
import unittest

from typing import Iterable

from ..src import create_interpolant
from ..utils import InputData, safe_broadcast, safe_cat

class ErrorMessages:
    """
    Class containing the error messages for the tests
    """
    test_shape = lambda input_shape, target_shape: f"Input shape ({input_shape}) is different from Target shape ({target_shape})"
    test_equal = lambda input, target: f"The input tensor and the target tensor differs: input ({input}), target ({target})"

class Test(unittest.TestCase):
    """
    Class for handling tests
    """
    def check_shape(self, input: InputData, target_shape: Iterable) -> None:
        """
        Function for checking the shape of the input tensor against the expected shape
        """
        msg = ErrorMessages.test_shape
        input_shape = input.shape
        self.assertSequenceEqual(input_shape, target_shape, msg(input_shape, target_shape))
    
    def check_equal(self, input: InputData, target: InputData) -> None:
        """
        Function for checking if the `input` tensor matches the `target`
        """
        msg = ErrorMessages.test_equal
        cond = torch.all(torch.eq(input, target))
        self.assertTrue(cond, msg(input, target))
    
    def test_safe_cat(self) -> None:
        """
        Tests the function `safe_cat` from the `common` module
        """
        ## defining batch of noisy images and time indices
        x0 = torch.randn((16, 2))
        x1 = torch.randn((16, 2))
        xc = torch.randn((16, 2))
        xs = torch.randn((16, 2))
        t = torch.rand((16))
        batch = {"x0": x0, "x1": x1, "xc": xc, "t": t, "xs": xs}
        ## defining keys to concatenate and broadcast
        cat_keys = ["t", "xc", "xs"]
        to_broadcast = ["t"]
        ## performing safe cat (only unsqueezing)
        xcat = safe_cat(batch, cat_keys, to_broadcast)
        expected_shape = torch.Size((16, 5))
        ## performing safe cat (broadcasting)
        xcat = safe_cat(batch, cat_keys, to_broadcast, unsqueeze_last_dim = False)
        expected_shape = torch.Size((16, 6))
        self.check_shape(xcat, expected_shape)

    def test_safe_broadcast(self) -> None:
        """
        Tests the function `safe_broadcast` from the `common` module
        """
        ...