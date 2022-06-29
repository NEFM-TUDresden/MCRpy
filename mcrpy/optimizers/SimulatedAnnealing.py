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
from typing import Tuple

import numpy as np
import tensorflow as tf

from mcrpy.optimizers.Optimizer import Optimizer
from mcrpy.src import optimizer_factory

class SimulatedAnnealing(Optimizer):
    is_gradient_based = False
    is_vf_based = True

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
            **kwargs):
        self.max_iter = max_iter
        self.conv_iter = conv_iter
        self.reconstruction_callback = callback
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature if cooldown_factor is None else -0.1
        self.cooldown_factor = cooldown_factor if cooldown_factor is not None else (
                final_temperature / initial_temperature) ** (1.0 / (max_iter - 1))
        if not use_multiphase:
            raise NotImplementedError('SimulatedAnnealing needs multiphase')

        self.tolerance = tolerance
        self.current_loss = np.inf
        self.loss = loss
        self.volume_fractions = None

        assert self.reconstruction_callback is not None
        assert self.loss is not None

    def set_volume_fractions(self, volume_fractions: np.ndarray):
        self.volume_fractions = volume_fractions

    @staticmethod
    def acceptance_distribution(temperature: float, delta_loss: float):
        acceptance = np.exp(delta_loss / temperature)
        return acceptance

    @staticmethod
    def mutation(x: tf.Tensor, n_attempts: int = 1000) -> np.ndarray:
        """Perform a random mutation while keeping the volume fraction constant."""
        mutant = np.copy(x.numpy())
        index_1 = tuple([random.randint(0, s_i - 1) for s_i in mutant.shape])[:-1]
        phases_1 = mutant[index_1]
        for n_attempt in range(n_attempts):
            index_2 = tuple([random.randint(0, s_i - 1) for s_i in mutant.shape])[:-1]
            phases_2 = mutant[index_2]
            if np.sum(np.square(phases_1 - phases_2)) > 1e-9:
                break
        else:
            logging.warning('reached max number of mutation attempts')
        mutant[index_1] = phases_2
        mutant[index_2] = phases_1
        return mutant
            
    @staticmethod
    def fix_initialization(x: tf.Tensor, volume_fractions: np.ndarray):
        """Rounds values and makes sure the volume fractions are met."""
        def get_state_probability(field: np.ndarray, state: int) -> float:
            """Get the probability of state."""
            indicator_function = field[..., int(round(state))]
            state_probability = np.sum(indicator_function) / indicator_function.size
            return state_probability

        def sample_phase_location(field: np.ndarray, phase: int) -> Tuple(int):
            possible_locations = np.where(field[..., int(round(phase))] == 1)
            n_possibilities = len(possible_locations[0])
            if not n_possibilities:
                return tuple([random.randint(0, s_i - 1) for s_i in field.shape[:-1]])
            chosen_possibility = random.randint(0, n_possibilities - 1)
            return tuple([location_1d[chosen_possibility] for location_1d in possible_locations])

        logging.info('start fix_initialization')
        if isinstance(volume_fractions, tuple):
            volume_fractions = volume_fractions[0]
        volume_fractions = volume_fractions[:, 0].flatten()
        field = np.round(np.copy(x.numpy()), decimals=0).astype(np.int32)
        init_fails = np.where(np.sum(field, axis=-1) != 1)
        for init_fail in zip(*init_fails):
            field[init_fail] = 0
            field[init_fail][0] = 1
        previous_loss = np.inf
        n_phases = len(volume_fractions)
        while True:
            state_probabilities = np.array([get_state_probability(field, state) 
                for state in range(n_phases)])
            vf_delta = state_probabilities - volume_fractions
            new_loss = np.sum((vf_delta)**2)
            if new_loss >= previous_loss:
                break
            phase_over = np.where(vf_delta == max(vf_delta))[0][0]
            phase_under = np.where(vf_delta == min(vf_delta))[0][0]
            place_over = sample_phase_location(field, phase_over)
            field[place_over][phase_over] = 0
            field[place_over][phase_under] = 1
            previous_loss = new_loss
        x.assign(field)
        logging.info('done fix_initialization')
        return x

    def step(self):
        """Perform one step."""
        self.x_mutated.assign(self.mutation(self.x))
        new_loss = self.call_loss(self.x_mutated)
        loss_amelioration = self.current_loss - new_loss

        if loss_amelioration > 0 or random.uniform(0, 1) < self.acceptance_distribution( 
                self.temperature, loss_amelioration):
            self.x.assign(self.x_mutated)
            self.current_loss = new_loss
            self.iters_since_last_accept = 0
        else:
            self.iters_since_last_accept += 1

        self.temperature *= self.cooldown_factor
        self.n_iter += 1
        self.reconstruction_callback(self.n_iter, self.current_loss, self.x)

    def optimize(self, x: tf.Tensor) -> int:
        """Optimization loop."""
        if self.volume_fractions is None:
            raise ValueError('volume_fractions are None, maybe not assigned?')
        self.n_iter = 0
        self.iters_since_last_accept = 0
        self.x = self.fix_initialization(x, self.volume_fractions)
        self.x_mutated = tf.Variable(initial_value=self.x, trainable=False)
        self.n_phases = self.x.numpy().shape[-1]

        self.current_loss = self.call_loss(self.x)
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
