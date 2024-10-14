import unittest

import numpy as np

class TestNorms(unittest.TestCase):
    def test_norms(self):
        from mcrpy.losses.L1 import L1
        from mcrpy.losses.L2 import L2
        from mcrpy.losses.MSE import MSE
        from mcrpy.losses.RMS import RMS
        from mcrpy.losses.SSE import SSE
        my_vector = np.array([1.0, 2.0, 3.0])
        for norm_type, expected_result in zip(
                [L1, L2, MSE, RMS, SSE],
                [6.0, np.sqrt(14), 14/3, np.sqrt(14/3), 14]
            ):
            norm_func = norm_type.define_norm()
            computed_result = norm_func(my_vector).numpy()
            self.assertAlmostEqual(computed_result, expected_result)
        