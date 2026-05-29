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
from mcrpy.descriptors.Descriptor import make_image_padder
from mcrpy.descriptors.OrientationDescriptor import OrientationDescriptor


class ODFExpansion(OrientationDescriptor):
    """Orientation distribution function by spatial average of each term of SHSH series expansion"""

    is_differentiable = True
    default_weight = 1000.0

    @staticmethod
    def make_orientation_descriptor(n_shsh_terms: int = None, **kwargs) -> callable:

        @tf.function
        def model(x):
            descriptors = []
            for o_dim in range(n_shsh_terms):
                mg_input = x[:, :, :, o_dim]
                odf_coeff = tf.reduce_mean(mg_input)
                descriptors.append(odf_coeff)
            return tf.stack(descriptors, axis=0)

        return model


def register() -> None:
    descriptor_factory.register("ODFExpansion", ODFExpansion)
