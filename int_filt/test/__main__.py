"""
This file runs all the necessary tests
"""
import unittest

from .common import Test
from .test_surr_si_interpolant import SurrSIInterpolantTest
from .test_surr_si_objectives import SurrSIObjectivesTest

def create_test_suite(test_case_ids: dict[str, unittest.TestCase], test_ids: dict[str, str]) -> unittest.TestSuite:
    """
    Function for allocating test suite
    """
    ## initializing suite
    suite = unittest.TestSuite()
    ## adding all required tests
    for test_dict in test_ids:
        for test_case_id, test_id in test_dict.items():
            test_case = test_case_ids[test_case_id]
            test = test_case(test_id)
            suite.addTest(test)
    return suite

if __name__ == "__main__":
    ## define here the test cases ids
    test_case_ids = {
        "common": Test,
        "surr_si_intrpl": SurrSIInterpolantTest,
        "surr_si_objctv": SurrSIObjectivesTest
    }
    ## define here the tests to run
    test_ids = [
        ## common interfaces
        {"common": "test_safe_cat"},
        ## surr-si pffp-v0 interpolant
        {"surr_si_intrpl": "test_surr_si_intrpl_pffp_v0_coeffs_alpha"}, 
        {"surr_si_intrpl": "test_surr_si_intrpl_pffp_v0_coeffs_beta"}, 
        {"surr_si_intrpl": "test_surr_si_intrpl_pffp_v0_coeffs_sigma"},
        {"surr_si_intrpl": "test_surr_si_intrpl_pffp_v0_coeffs_alpha_dot"}, 
        {"surr_si_intrpl": "test_surr_si_intrpl_pffp_v0_coeffs_beta_dot"}, 
        {"surr_si_intrpl": "test_surr_si_intrpl_pffp_v0_coeffs_sigma_dot"},
        ## surr-si pffp-v1 interpolant
        {"surr_si_intrpl": "test_surr_si_intrpl_pffp_v1_coeffs_alpha"}, 
        {"surr_si_intrpl": "test_surr_si_intrpl_pffp_v1_coeffs_beta"}, 
        {"surr_si_intrpl": "test_surr_si_intrpl_pffp_v1_coeffs_sigma"},
        {"surr_si_intrpl": "test_surr_si_intrpl_pffp_v1_coeffs_alpha_dot"}, 
        {"surr_si_intrpl": "test_surr_si_intrpl_pffp_v1_coeffs_beta_dot"}, 
        {"surr_si_intrpl": "test_surr_si_intrpl_pffp_v1_coeffs_sigma_dot"},
        ## surr-si drift objective 
        {"surr_si_objctv": "test_surr_si_objctv_drift"}, 
    ]
    ## defining suite from test cases
    suite = create_test_suite(test_case_ids, test_ids)
    ## defining test runnner
    runner = unittest.TextTestRunner(verbosity=2)
    ## running tests
    runner.run(suite)