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
from mcrpy.descriptors.Descriptor import make_image_padder


class Variation(PhaseDescriptor):
    is_differentiable = True
    default_weight = 100.0

    @staticmethod
    def make_singlephase_descriptor(
            **kwargs) -> callable:
        """Make a periodic variation descriptor. This is not the total variation but the
        mean variation, since the total variation would not be a proper descriptor to compare
        across different resolutions.""" 

        tile_img = make_image_padder(1, 1)

        @tf.function
        def periodic_variation(img: tf.Tensor) -> tf.Tensor:
            img_tiled = tile_img(img)
            var = tf.image.total_variation(img_tiled) / tf.cast(tf.math.reduce_prod(tf.shape(img)), tf.float64)
            return var
        return periodic_variation

def register() -> None:
    descriptor_factory.register("Variation", Variation)
