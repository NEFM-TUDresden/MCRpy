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

import tensorflow as tf

from mcrpy.optimizers.TFOptimizer import TFOptimizer
from mcrpy.src import optimizer_factory


class Adamax(TFOptimizer):
    def __init__(self,
                 max_iter: int = 100,
                 callback: callable = None,
                 learning_rate: float = 0.001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 **kwargs):
        super().__init__(max_iter=max_iter, callback=callback)
        self.opt = tf.keras.optimizers.Adamax(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)


def register() -> None:
    optimizer_factory.register("Adamax", Adamax)
