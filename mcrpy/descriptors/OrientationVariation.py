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
from typing import Callable

import tensorflow as tf

from mcrpy.src import descriptor_factory
from mcrpy.descriptors.OrientationDescriptor import OrientationDescriptor
from mcrpy.descriptors.Descriptor import make_image_padder


class OrientationVariation(OrientationDescriptor):
    """Variation for each dimension of orientation"""

    is_differentiable = True
    default_weight = 1.0
    required_orientation_representation = None  # none means SHSH
    orientation_representation_dimension: int = 9

    @classmethod
    def make_orientation_descriptor(
        cls,
        **kwargs,
    ) -> Callable:
        """Make a periodic variation descriptor. This is not the total variation but the
        mean variation, since the total variation would not be a proper descriptor to compare
        across different resolutions."""

        tile_img = make_image_padder(1, 1)

        @tf.function
        def periodic_variation(img: tf.Tensor) -> tf.Tensor:
            img_tiled = tile_img(tf.expand_dims(img, axis=-1))
            var = tf.image.total_variation(img_tiled) / tf.cast(tf.math.reduce_prod(tf.shape(img)), tf.float64)
            return var

        @tf.function
        def model(x: tf.Tensor) -> tf.Tensor:
            all_distributions = [
                periodic_variation(x[:, :, :, i]) for i in range(cls.orientation_representation_dimension)
            ]
            return tf.concat(all_distributions, 0)

        return model


def register() -> None:
    descriptor_factory.register("OrientationVariation", OrientationVariation)
