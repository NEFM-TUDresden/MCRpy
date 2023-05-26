from contextlib import contextmanager
import itertools
import logging
import pickle
import random
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.ndimage import convolve
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from mcrpy.src.Symmetry import Symmetry, Cubic
from mcrpy.src.Microstructure import Microstructure

class MutableMicrostructure(Microstructure):

    def __init__(self, 
            array: np.ndarray, 
            use_multiphase = False,
            ori_repr: type = None,
            symmetry: Symmetry = Cubic,
            skip_encoding: bool = False,
            trainable: bool = True):
        super().__init__(
            array,
            use_multiphase = use_multiphase,
            ori_repr = ori_repr,
            symmetry = symmetry,
            skip_encoding = skip_encoding,
            trainable = trainable)

        self.linear_indices = np.array(list(range(np.product(self.spatial_shape))))
        if self.is_3D:
            self.conv_weights = np.array([[[0, 0,  0], [0, -1, 0], [0, 0, 0]], [[0, -1, 0], [-1, 6, -1], [0, -1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 0]]], dtype=np.int32)
            self.neighbor_conv_weights = np.array([[[0, 0,  0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=np.int32)
        else:
            self.conv_weights = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.int32)
            self.neighbor_conv_weights = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32)
        self.pre_mutation = None
        self.post_mutation = None


    def __repr__(self):
        representation = super().__repr__()
        if self.pre_mutation is not None:
            representation += """
            Pre-mutation version is available."""
        if self.post_mutation is not None:
            representation += """
            Post-mutation version is available."""
            return representation

    def sample_phase_location(
            self,
            field: np.ndarray,
            phase: int,
            neighbor_offset: float = 2.9,
            multiple: int = None,
            neighbor_in_phase: Union[int, None] = None):
        conv_result = convolve(field[0,...,phase], self.conv_weights, mode='wrap').flatten()
        if neighbor_in_phase is not None:
            conv_result = conv_result * convolve(
                    field[0, ..., neighbor_in_phase],
                    self.neighbor_conv_weights,
                    mode='wrap').flatten()
        clipped_result = np.clip(conv_result-neighbor_offset, 0, 10)
        unscaled_probs = clipped_result**2
        assert np.sum(unscaled_probs>0)>0 , "no pixel remaining"
        probs = unscaled_probs / np.sum(unscaled_probs)
        chosen_linear_index = np.random.choice(self.linear_indices, multiple, replace=False, p=probs)
        if multiple in {0, 1, None}:
            return np.unravel_index(chosen_linear_index, field.shape[:-1])
        else:
            return [np.unravel_index(cli, field.shape[:-1]) for cli in chosen_linear_index]

    def mutate(self, n_attempts: int = 1000, rule: str = 'relaxed_neighbor'):
        self.pre_mutation = tf.identity(self.x)
        with self.use_multiphase_encoding() as x:
            mutant = x.numpy()
            if rule == 'random':
                index_1 = tuple([0] + [random.randint(0, s_i - 1) for s_i in self.spatial_shape])
                phases_1 = np.argmax(mutant[index_1])
                for n_attempt in range(n_attempts):
                    index_2 = tuple([0] + [random.randint(0, s_i - 1) for s_i in self.spatial_shape])
                    phases_2 = np.argmax(mutant[index_2])
                    if np.sum(np.square(phases_1 - phases_2)) > 1e-9:
                        break
                else:
                    logging.warning('reached max number of mutation attempts')
            elif rule == 'relaxed_neighbor':
                swap_phases = random.sample(list(range(max(2, self.n_phases))), 2)
                swap_indices = [self.sample_phase_location(mutant, phase, neighbor_offset=0) for phase in swap_phases]
                index_1, index_2 = swap_indices
                phases_1, phases_2 = swap_phases
            elif rule == 'neighbor':
                swap_phases = random.sample(list(range(max(2, self.n_phases))), 2)
                swap_indices = [self.sample_phase_location(mutant, phase, neighbor_offset=2.9 if self.is_3D else 1.9) for phase in swap_phases]
                index_1, index_2 = swap_indices
                phases_1, phases_2 = swap_phases
            elif rule == 'neighbor_in_phase':
                swap_phases = random.sample(list(range(max(2, self.n_phases))), 2)
                phases_1, phases_2 = swap_phases
                neighbor_offset=2.9 if self.is_3D else 1.9
                index_1 = self.sample_phase_location(
                        mutant, phases_1, neighbor_offset=neighbor_offset, 
                        neighbor_in_phase=phases_2)
                index_2 = self.sample_phase_location(
                        mutant, phases_2, neighbor_offset=neighbor_offset, 
                        neighbor_in_phase=phases_1)
            else:
                raise NotImplementedError(f'Mutation rule {rule} is not implemented')
            mutant[index_1][phases_1] = 0
            mutant[index_1][phases_2] = 1
            mutant[index_2][phases_2] = 0
            mutant[index_2][phases_1] = 1
            if self.is_3D:
                self.swapped_index_1.assign(np.array(index_1[1:], dtype=np.int32))
                self.swapped_index_2.assign(np.array(index_2[1:], dtype=np.int32))
            x.assign(mutant)
            
    def redo_mutation(self):
        assert self.post_mutation is not None
        self.x.assign(self.post_mutation)

    def undo_mutation(self):
        assert self.pre_mutation is not None
        self.post_mutation = tf.identity(self.x)
        self.x.assign(self.pre_mutation)

    def adjust_vf(self, volume_fractions: np.ndarray):
        """Rounds values and makes sure the volume fractions are met."""
        def get_state_probability(field: np.ndarray, state: int) -> float:
            """Get the probability of state."""
            indicator_function = field[..., int(round(state))]
            state_probability = np.sum(indicator_function) / indicator_function.size
            return state_probability

        # def sample_phase_location(field: np.ndarray, phase: int) -> Tuple[int]:
        #     possible_locations = np.where(field[..., int(round(phase))] == 1)
        #     n_possibilities = len(possible_locations[0])
        #     if not n_possibilities:
        #         return tuple([random.randint(0, s_i - 1) for s_i in field.shape[:-1]])
        #     chosen_possibility = random.randint(0, n_possibilities - 1)
        #     return tuple([location_1d[chosen_possibility] for location_1d in possible_locations])

        logging.info('start adjusting volume fractions')
        neighbor_offset = 2.9
        with self.use_multiphase_encoding() as x:
            if isinstance(volume_fractions, tuple):
                volume_fractions = volume_fractions[0]
            volume_fractions = volume_fractions[:, 0].flatten()
            if len(volume_fractions) == 1:
                volume_fractions = np.array([1-volume_fractions[0], volume_fractions[0]])
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
                excess = int(max(vf_delta[phase_over], np.abs(vf_delta[phase_under])) * np.product(self.spatial_shape) / 10)
                multiple = excess if excess >= 2 else None
                # multiple = 10 if vf_delta[phase_over] * np.product(self.spatial_shape) > 10 else None
                print(f'{volume_fractions} - {state_probabilities} = {vf_delta} - loss {new_loss} - {phase_under} from {phase_over}')
                if neighbor_offset < 0:
                    raise ValueError
                try:
                    place_over = self.sample_phase_location(field, phase_over, neighbor_offset=neighbor_offset, multiple=multiple, neighbor_in_phase=phase_under)
                except (AssertionError, ValueError):
                    neighbor_offset -= 0.5
                    continue
                if multiple in {0, 1, None}:
                    field[place_over][phase_over] = 0
                    field[place_over][phase_under] = 1
                else:
                    for p_o in place_over:
                        field[p_o][phase_over] = 0
                        field[p_o][phase_under] = 1
                previous_loss = new_loss
            x.assign(field)
        logging.info('done adjusting volume fractions')
