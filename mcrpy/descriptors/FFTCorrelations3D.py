from __future__ import annotations
from typing import Tuple

import numpy as np
import tensorflow as tf

from mcrpy.src import descriptor_factory
from mcrpy.descriptors.PhaseDescriptor3D import PhaseDescriptor3D


class FFTCorrelations3D(PhaseDescriptor3D):
    """2-point autocorrelations in 3D calculated by FFT"""

    is_differentiable = True

    @staticmethod
    def make_singlephase_descriptor(desired_shape_2d=(64, 64), limit_to: int = None, **kwargs) -> callable:

        assert desired_shape_2d[0] == desired_shape_2d[1]
        mask = np.zeros((desired_shape_2d[0],) * 3)
        i_center = desired_shape_2d[0] // 2
        mask[
            i_center - limit_to + 1 : i_center + limit_to,
            i_center - limit_to + 1 : i_center + limit_to,
            i_center - limit_to + 1 : i_center + limit_to,
        ] = 1
        mask = mask.astype(bool)
        mask = tf.constant(mask, dtype=tf.bool)
        descriptor_shape = (limit_to * 2 - 1,) * 3

        @tf.function
        def compute_descriptor(microstructure: tf.Tensor) -> tf.Tensor:
            # tf.print(tf.shape(microstructure))
            microstructure = tf.reshape(microstructure, (desired_shape_2d[0],) * 3)
            ms_fourier = tf.signal.fft3d(tf.cast(microstructure, tf.complex128))
            ms_conj = tf.math.conj(ms_fourier)
            fourier_coefficients = ms_fourier * ms_conj / tf.cast(tf.size(microstructure), tf.complex128)
            real_coefficients = tf.cast(tf.signal.ifft3d(fourier_coefficients), tf.float64)
            real_coefficients = tf.signal.fftshift(real_coefficients)
            selected_components = tf.reshape(
                tf.boolean_mask(tf.cast(real_coefficients, tf.float64), mask), descriptor_shape
            )
            return selected_components

        return compute_descriptor


def register() -> None:
    descriptor_factory.register("FFTCorrelations3D", FFTCorrelations3D)
