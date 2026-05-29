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
import logging

import numpy as np
import tensorflow as tf

from mcrpy.descriptors.PhaseDescriptor import PhaseDescriptor
from mcrpy.src.IndicatorFunction import IndicatorFunction


class PhaseDescriptor3D(PhaseDescriptor):

    @classmethod
    def make_descriptor(
        cls,
        full_3d: bool = False,
        use_multiphase: bool = True,
        use_multigrid_descriptor=True,
        limit_to=8,
        desired_shape_2d=None,
        desired_shape_extended=None,
        **kwargs,
    ) -> callable:
        """By default wraps self.make_single_phase_descriptor."""
        assert full_3d
        n_phases = desired_shape_extended[-1]

        if use_multigrid_descriptor:
            singlephase_descriptor = cls.make_multigrid_descriptor(
                limit_to=limit_to,
                desired_shape_2d=desired_shape_2d,
                desired_shape_extended=desired_shape_extended,
                full_3d=full_3d,
                **kwargs,
            )
        else:
            singlephase_descriptor = cls.make_singlegrid_descriptor(
                limit_to=limit_to,
                desired_shape_2d=desired_shape_2d,
                desired_shape_extended=desired_shape_extended,
                full_3d=full_3d,
                **kwargs,
            )

        @tf.function
        def singlephase_wrapper(x: tf.Tensor) -> tf.Tensor:
            phase_descriptor = tf.expand_dims(singlephase_descriptor(x), axis=0)
            return phase_descriptor

        @tf.function
        def multiphase_wrapper(x: tf.Tensor) -> tf.Tensor:
            phase_descriptors = []
            for phase in range(n_phases):
                x_phase = x[:, :, :, :, phase]
                phase_descriptor = tf.expand_dims(singlephase_descriptor(tf.expand_dims(x_phase, axis=-1)), axis=0)
                phase_descriptors.append(phase_descriptor)
            return tf.concat(phase_descriptors, axis=0)

        return multiphase_wrapper if use_multiphase else singlephase_wrapper

    @classmethod
    def make_multigrid_descriptor(cls, desired_shape_2d=None, desired_shape_extended=None, limit_to=8, **kwargs):

        H, W = desired_shape_2d
        n_phases = desired_shape_extended[-1]
        limitation_factor = min(H / limit_to, W / limit_to)
        mg_levels = int(np.floor(np.log(limitation_factor) / np.log(2)))
        singlephase_descriptors = []
        for mg_level in range(mg_levels):
            pool_size = 2**mg_level
            if H % pool_size != 0 or W % pool_size != 0:
                logging.warning("For MG level number {}, an avgpooling remainder exists.".format(mg_level))
            desired_shape_layer = tuple(s // pool_size for s in desired_shape_2d)
            singlephase_kwargs = {
                "desired_shape_2d": desired_shape_layer,
                "desired_shape_extended": (1, *desired_shape_layer, n_phases),
                "limit_to": limit_to,
                **kwargs,
            }
            singlephase_descriptors.append(cls.wrap_singlephase_descriptor(**singlephase_kwargs))
        else:
            logging.info("Could create all {} MG levels".format(mg_levels))

        def multigrid_descriptor(mg_input):
            mg_layers = []
            for mg_level, singlephase_descriptor in enumerate(singlephase_descriptors):
                pool_size = 2**mg_level
                if isinstance(mg_input, IndicatorFunction):
                    mg_pool = IndicatorFunction(avg_pool3d(mg_input.x, pool_size))
                else:
                    mg_pool = avg_pool3d(mg_input, pool_size)
                mg_desc = singlephase_descriptor(mg_pool)
                mg_exp = tf.expand_dims(mg_desc, axis=0)
                mg_layers.append(mg_exp)
            outputs = tf.concat(mg_layers, axis=0) if len(mg_layers) > 1 else mg_layers[0]
            return outputs

        return multigrid_descriptor


def avg_pool3d(x: tf.Tensor, pool_size: int):
    if pool_size < 2:
        return x
    y = (
        x[:, ::2, ::2, ::2, :]
        + x[:, 1::2, ::2, ::2, :]
        + x[:, ::2, 1::2, ::2, :]
        + x[:, ::2, ::2, 1::2, :]
        + x[:, 1::2, 1::2, ::2, :]
        + x[:, 1::2, ::2, 1::2, :]
        + x[:, ::2, 1::2, 1::2, :]
        + x[:, 1::2, 1::2, 1::2, :]
    ) / 8.0
    if pool_size > 2:
        y = avg_pool3d(y, pool_size=pool_size // 2)
    return y
