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

from mcrpy.src.profile import maybe_trace

from mcrpy.optimizers.Optimizer import Optimizer
from mcrpy.src.Microstructure import Microstructure

class TFOptimizer(Optimizer):
    is_gradient_based = True
    is_vf_based = False
    is_sparse = False

    def __init__(self,
            max_iter: int = 100,
            callback: callable = None):
        """ABC for TensorFlow optimizers. Subclasses only have to initlialize self.opt."""
        self.max_iter = max_iter
        self.reconstruction_callback = callback
        self.current_loss = None

        assert self.reconstruction_callback is not None

    @maybe_trace("step")
    def step(self):
        """Perform one step."""
        loss, grads = self.call_loss(self.ms)
        self.current_loss = loss
        self.reconstruction_callback(self.n_iter, loss, self.ms)
        self.opt.apply_gradients(zip(grads, self.opt_var))
        self.n_iter += 1

    def optimize(self, ms: Microstructure, restart_from_niter: int = None) -> int:
        """Optimization loop."""
        self.n_iter = 0 if restart_from_niter is None else restart_from_niter
        self.ms = ms
        self.opt_var = [ms.x]
        while self.n_iter < self.max_iter:
            self.step()
        return self.n_iter
