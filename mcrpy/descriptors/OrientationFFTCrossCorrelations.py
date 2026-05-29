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
import tensorflow as tf
import numpy as np

from mcrpy.src import descriptor_factory
from mcrpy.descriptors.OrientationDescriptor import OrientationDescriptor


class OrientationFFTCrossCorrelations(OrientationDescriptor):
    """2-point cross-correlation of all dimensions of orientation field calculated by FFT"""

    is_differentiable = True
    default_weight = 10.0
    required_orientation_representation = None  # none means SHSH

    @classmethod
    def make_orientation_descriptor(
        cls, desired_shape_2d=(64, 64), limit_to: int = 8, n_shsh_terms: int = None, **kwargs
    ) -> callable:

        include_auto = False
        mask = np.zeros(desired_shape_2d)
        i_center = desired_shape_2d[0] // 2
        j_center = desired_shape_2d[1] // 2
        mask[i_center - limit_to + 1 : i_center + limit_to, j_center - limit_to + 1 : j_center + limit_to] = 1
        mask = mask.astype(bool)
        mask = tf.constant(mask, dtype=tf.bool)
        descriptor_shape = (limit_to * 2 - 1,) * 2

        @tf.function
        def model(mg_input):
            correlation_list = []
            for phase_from in range(n_shsh_terms):
                for phase_to in range(phase_from + 1):
                    if phase_from == phase_to and not include_auto:
                        continue
                    ms_from = tf.cast(mg_input[0, :, :, phase_from], tf.complex128)
                    ms_from_fourier = tf.signal.fft2d(ms_from)
                    ms_to = tf.cast(mg_input[0, :, :, phase_to], tf.complex128)
                    ms_to_fourier = tf.signal.fft2d(ms_to)
                    fourier_coefficients = (
                        ms_from_fourier * tf.math.conj(ms_to_fourier) / tf.cast(tf.size(ms_from), tf.complex128)
                    )
                    real_coefficients = tf.signal.ifft2d(fourier_coefficients)
                    real_coefficients = tf.signal.fftshift(real_coefficients)
                    selected_components = tf.reshape(
                        tf.boolean_mask(tf.cast(real_coefficients, tf.float64), mask), descriptor_shape
                    )
                    correlation_list.append(selected_components)
            return tf.stack(correlation_list, axis=0)

        return model


def register() -> None:
    descriptor_factory.register("OrientationFFTCrossCorrelations", OrientationFFTCrossCorrelations)
