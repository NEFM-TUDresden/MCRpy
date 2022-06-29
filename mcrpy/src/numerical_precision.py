"""
   Copyright 10/2020 - 04/2021 Paul Seibert for Diploma Thesis at TU Dresden
   Copyright 05/2021 - 12/2021 TU Dresden (Paul Seibert as Scientific Assistant)
   Copyright 2022 TU Dresden (Paul Seibert as Scientific Employee)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from typing import Any

def set_precision(obj: Any, precision: int):
    """Set Numpy and TensorFlow precision of obj and Keras floatx backend."""
    if precision == 32:
        obj.np_dtype = np.float32
        obj.tf_dtype = tf.float32
        tf.keras.backend.set_floatx('float32')
    elif precision == 64:
        obj.np_dtype = np.float64
        obj.tf_dtype = tf.float64
        tf.keras.backend.set_floatx("float64")
    else:
        raise NotImplementedError()

