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
from scipy.sparse import csr_matrix

import scipy.optimize as sopt
from mcrpy.optimizers.SPOptimizer import SPOptimizer
from mcrpy.src import optimizer_factory


class LBFGSB(SPOptimizer):
    def __init__(self,
                 max_iter: int = 100,
                 callback: callable = None,
                 desired_shape_extended: tuple = None,
                 np_dtype: np.dtype = None,
                 **kwargs):

        super().__init__(max_iter=max_iter, callback=callback, desired_shape_extended=desired_shape_extended, np_dtype=np_dtype)
        self.optimizer_method = 'L-BFGS-B'
        bounds_shape = (np.product(desired_shape_extended), 2)
        self.bounds = np.zeros(bounds_shape)
        self.bounds[:, 0] = 0
        self.bounds[:, 1] = 1


def register() -> None:
    optimizer_factory.register("LBFGSB", LBFGSB)
