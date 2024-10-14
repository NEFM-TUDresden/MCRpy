import unittest
from unittest.mock import Mock

from mcrpy.src.numerical_precision import set_precision

import numpy as np
import tensorflow as tf

class TestPrecision(unittest.TestCase):
    def test_argumments(self):
        for unsupported_input in [
            None, 8, 16, 128, "8", "16", "32", "64", "128"
        ]:
            with self.assertRaises(NotImplementedError):
                set_precision(None, unsupported_input)
    
    def test_effect_32(self):
        mock = Mock()
        set_precision(mock, 32)
        self.assertEqual(mock.np_dtype, np.float32)
        self.assertEqual(mock.tf_dtype, tf.float32)
        self.assertEqual(tf.keras.backend.floatx(), 'float32')
    
    def test_effect_64(self):
        mock = Mock()
        set_precision(mock, 64)
        self.assertEqual(mock.np_dtype, np.float64)
        self.assertEqual(mock.tf_dtype, tf.float64)
        self.assertEqual(tf.keras.backend.floatx(), 'float64')