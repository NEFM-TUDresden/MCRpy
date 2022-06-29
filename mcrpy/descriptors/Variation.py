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
from mcrpy.descriptors.Descriptor import Descriptor


class Variation(Descriptor):
    is_differentiable = True

    @staticmethod
    def make_singlephase_descriptor(
            tf_dtype: tf.DType = tf.float32, 
            **kwargs) -> callable:
        """Make a periodic variation descriptor. This is not the total variation but the
        mean variation, since the total variation would not be a proper descriptor to compare
        across different resolutions.""" 

        @tf.function
        def tile_img(img: tf.Tensor) -> tf.Tensor:
            """Tile an image. Needed for periodic boundary conditions in variation."""
            _, h, w, _ = img.shape.as_list()
            img_shape_2d = (h, w)

            tiled_shape = np.array(img_shape_2d) + np.array([1, 1])
            h_desired = h + 1
            w_desired = w + 1

            pre_tiler = np.zeros((h_desired, h), dtype=np.int8)
            for i in range(h_desired):
                pre_tiler[i % h_desired, i % h] = 1
            pre_tiler_tf = tf.cast(tf.constant(pre_tiler), tf_dtype)
            post_tiler = np.zeros((w, w_desired), dtype=np.int8)
            for i in range(w_desired):
                post_tiler[i % w, i % w_desired] = 1
            post_tiler_tf = tf.cast(tf.constant(post_tiler), tf_dtype)
            img_tiled = tf.reshape(tf.linalg.matmul(pre_tiler_tf, tf.linalg.matmul(
                tf.reshape(img, img_shape_2d), post_tiler_tf)), [1] + [*tiled_shape] + [1])
            return img_tiled

        @tf.function
        def periodic_variation(img: tf.Tensor) -> tf.Tensor:
            img_tiled = tile_img(img)
            var = tf.image.total_variation(img_tiled) / tf.cast(tf.math.reduce_prod(tf.shape(img)), tf_dtype)
            return var
        return periodic_variation

def register() -> None:
    descriptor_factory.register("Variation", Variation)
