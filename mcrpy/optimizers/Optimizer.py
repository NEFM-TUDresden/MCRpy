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
from abc import ABC, abstractmethod

import tensorflow as tf


class Optimizer(ABC):
    """Basic representation of an optimizer."""
    current_loss = None
    is_gradient_based = True
    is_vf_based = False

    @abstractmethod
    def optimize(self, x: tf.Tensor):
        """Run optimization."""
        pass

    def set_call_loss(self, call_loss: callable):
        """Set function that calls the loss and handles the 2D/3D permutation
        and the gradient computation."""
        self.call_loss = call_loss

