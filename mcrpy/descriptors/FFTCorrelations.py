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
from typing import Tuple

import numpy as np
import tensorflow as tf

from mcrpy.src import descriptor_factory
from mcrpy.descriptors.PhaseDescriptor import PhaseDescriptor


class FFTCorrelations(PhaseDescriptor):
    is_differentiable = True

    @staticmethod
    def make_singlephase_descriptor(
            desired_shape_2d=(64, 64), 
            limit_to: int = None, 
            **kwargs) -> callable:

        mask = np.zeros(desired_shape_2d)
        i_center = desired_shape_2d[0] // 2
        j_center = desired_shape_2d[1] // 2
        mask[i_center - limit_to + 1:i_center+limit_to, j_center - limit_to + 1:j_center+limit_to] = 1
        mask = mask.astype(bool)
        mask = tf.constant(mask, dtype=tf.bool)
        descriptor_shape = (limit_to * 2 - 1,) *  2

        @tf.function
        def compute_descriptor(microstructure: tf.Tensor) -> tf.Tensor:
            microstructure = tf.reshape(microstructure, desired_shape_2d)
            ms_fourier = tf.signal.fft2d(tf.cast(microstructure, tf.complex128))
            ms_conj = tf.math.conj(ms_fourier)
            fourier_coefficients = ms_fourier * ms_conj / tf.cast(tf.size(microstructure), tf.complex128)
            real_coefficients = tf.signal.ifft2d(fourier_coefficients)
            real_coefficients = tf.signal.fftshift(real_coefficients)
            selected_components = tf.reshape(tf.boolean_mask(tf.cast(real_coefficients, tf.float64), mask), descriptor_shape)
            return selected_components
        return compute_descriptor

    @staticmethod
    def define_comparison_mask(
            desired_descriptor_shape: Tuple[int] = None, 
            limit_to: int = None, 
            **kwargs):
        assert len(desired_descriptor_shape) == 2
        assert desired_descriptor_shape[1] == desired_descriptor_shape[0]
        desired_limit_to = desired_descriptor_shape[1] // 2 + 1

        if limit_to == desired_limit_to:
            return None, False

        larger_limit_to = max(limit_to, desired_limit_to)
        smaller_limit_to = min(limit_to, desired_limit_to)
        limit_delta = larger_limit_to - smaller_limit_to
        larger_n_elements = larger_limit_to * 2 - 1
        mask = np.zeros((larger_n_elements, larger_n_elements), dtype=np.bool8)
        mask[limit_delta:-limit_delta, limit_delta:-limit_delta] = True
        return mask, limit_to > desired_limit_to

    @classmethod
    def visualize_subplot(
            cls,
            descriptor_value: np.ndarray,
            ax,
            descriptor_type: str = None,
            mg_level: int = None,
            n_phase: int = None):
        s2 = descriptor_value[:, :]
        height, width = s2.shape
        if height != width:
            raise NotImplementedError('Non-square FFTCorrelations not implemented')
        limit_to = height // 2 + 1
        ax.imshow(s2, cmap='cividis')
        ax.set_title(f'S2: l={mg_level}, p={n_phase}')
        ax.set_xlabel(r'$r_x$ in Px')
        ax.set_ylabel(r'$r_y$ in Px')
        xticks = [0, limit_to - 1, 2 * (limit_to - 1)]
        yticks = [0, limit_to - 1, 2 * (limit_to - 1)]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([-limit_to + 1, 0, limit_to - 1])
        ax.set_yticklabels(reversed([-limit_to + 1, 0, limit_to - 1]))


def register() -> None:
    descriptor_factory.register("FFTCorrelations", FFTCorrelations)
