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

import logging
import random

import numpy as np
import tensorflow as tf

from mcrpy.optimizers.Optimizer import Optimizer
from mcrpy.src import optimizer_factory
from mcrpy.src.MutableMicrostructure import MutableMicrostructure

class SimulatedAnnealing(Optimizer):
    is_gradient_based = False
    is_vf_based = True
    is_sparse = True
    swaps_pixels = True

    def __init__(self,
            max_iter: int = 100,
            conv_iter: int = 500,
            callback: callable = None,
            initial_temperature: float = 0.004,
            final_temperature: float = None,
            cooldown_factor: float = 0.99,
            tolerance: float = 0.00001,
            loss: callable = None,
            use_multiphase: bool = False,
            use_orientations: bool = False,
            is_3D: bool = False,
            mutation_rule: str = 'relaxed_neighbor',
            acceptance_distribution: str = 'zero_tolerance',
            **kwargs):
        if use_orientations:
            raise ValueError('This optimizer_type cannot solve for orientations.')
        self.max_iter = max_iter
        self.is_3D = is_3D
        self.conv_iter = conv_iter
        self.reconstruction_callback = callback
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature if cooldown_factor is None else -0.1
        self.cooldown_factor = cooldown_factor if cooldown_factor is not None else (
                final_temperature / initial_temperature) ** (1.0 / (max_iter - 1))

        self.tolerance = tolerance
        self.current_loss = np.inf
        self.loss = loss
        self.volume_fractions = None

        acceptance_distributions = {
                'zero_tolerance': self.zero_tolerance_acceptance_distribution,
                'exponential': self.exponential_acceptance_distribution,
                }
        if acceptance_distribution not in acceptance_distributions:
            raise ValueError(f'Acceptance distribution {acceptance_distribution} not supported')
        self.acceptance_distribution = acceptance_distributions[acceptance_distribution]
        self.mutation_rule = mutation_rule

        assert self.reconstruction_callback is not None
        assert self.loss is not None

    def set_volume_fractions(self, volume_fractions: np.ndarray):
        self.volume_fractions = volume_fractions

    @staticmethod
    def zero_tolerance_acceptance_distribution(temperature: float, delta_loss: float):
        return 0.0

    @staticmethod
    def exponential_acceptance_distribution(temperature: float, delta_loss: float):
        return np.exp(delta_loss / temperature)

    def step(self):
        """Perform one step."""
        self.ms.mutate(rule=self.mutation_rule)
        new_loss = self.call_loss(self.ms)
        if self.ms.is_3D:
            self.ms.undo_mutation()
            self.current_loss = self.call_loss(self.ms)
        loss_amelioration = self.current_loss - new_loss

        if loss_amelioration >= 0 or random.uniform(0, 1) < self.acceptance_distribution( 
                self.temperature, loss_amelioration):
            self.iters_since_last_accept = 0
            if self.ms.is_3D:
                self.ms.redo_mutation()
            else:
                self.current_loss = new_loss
        else:
            self.iters_since_last_accept += 1
            if self.ms.is_2D:
                self.ms.undo_mutation()

        self.temperature *= self.cooldown_factor
        self.n_iter += 1
        self.reconstruction_callback(self.n_iter, self.current_loss, self.ms)

    def optimize(self, ms: MutableMicrostructure, restart_from_niter: int = None) -> int:
        """Optimization loop."""
        if self.volume_fractions is None:
            raise ValueError('volume_fractions are None, maybe not assigned?')
        self.n_iter = 0 if restart_from_niter is None else restart_from_niter
        self.iters_since_last_accept = 0
        self.ms = ms
        self.ms.adjust_vf(self.volume_fractions)

        self.current_loss = self.call_loss(self.ms)
        self.temperature = self.initial_temperature
        while self.n_iter < self.max_iter:
            if self.temperature <= self.final_temperature:
                logging.info('reached temperature')
                break
            if self.iters_since_last_accept >= self.conv_iter:
                logging.info('converged - no change since {self.iters_since_last_accept} iterations')
                break
            if self.current_loss <= self.tolerance:
                logging.info('reached tolerance')
                break
            self.step()
        else:
            logging.info('reached number of iterations')
        return self.n_iter


def register() -> None:
    optimizer_factory.register("SimulatedAnnealing", SimulatedAnnealing)
