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


class OrientationVariogram(OrientationDescriptor):
    """3-point extension of variogram for orientations, similar to 3-point correlations"""

    is_differentiable = True
    default_weight = 10.0
    required_orientation_representation = None  # none means SHSH
    orientation_representation_dimension: int = 9

    @classmethod
    def make_orientation_descriptor(
        cls,
        desired_shape_2d=(64, 64),
        limit_to: int = 8,
        l_threshold_value: float = 0.75,
        threshold_steepness: float = 10,
        n_shsh_terms: int = None,
        periodic: bool = True,
        **kwargs,
    ) -> callable:
        H, W = desired_shape_2d
        H_conv = limit_to
        W_conv = limit_to
        z_lower_bound = tf.cast(
            1.0 / (1.0 + tf.math.exp(-((0.0 - l_threshold_value) * threshold_steepness))), dtype=tf.float64
        )
        z_upper_bound = tf.cast(
            1.0 / (1.0 + tf.math.exp(-((1.0 - l_threshold_value) * threshold_steepness))), dtype=tf.float64
        )
        a = tf.cast(1.0 / (z_upper_bound - z_lower_bound), dtype=tf.float64)
        b = tf.cast(-a * z_lower_bound, dtype=tf.float64)

        tile_img = make_image_padder(min(W_conv, W) - 1, min(H_conv, H) - 1)

        def make_dense_filters() -> tf.Tensor:
            in_channels = 1
            out_channels = H_conv * W_conv + (H_conv - 1) * (W_conv - 1)
            n_entries = out_channels * 2 - 1
            filter_indices = np.zeros((n_entries, 4), dtype=np.int64)
            filter_values = np.zeros((n_entries), dtype=np.float64)
            filter_denseshape = np.array([H_conv, W_conv, in_channels, out_channels], dtype=np.int64)
            entry_index = 0
            for i in range(H_conv):
                for j in range(W_conv):
                    k = i * W_conv + j

                    if i == 0 and j == 0:
                        filter_indices[entry_index] = (0, 0, 0, 0)
                        filter_values[entry_index] = 1
                        entry_index += 1
                        continue
                    filter_indices[entry_index] = (0, 0, 0, k)
                    filter_values[entry_index] = -1.0
                    entry_index += 1

                    filter_indices[entry_index] = (i, j, 0, k)
                    filter_values[entry_index] = 1.0
                    entry_index += 1
            for i in range(1, H_conv):
                for minus_j in range(1, W_conv):
                    j = -minus_j
                    k = H_conv * W_conv + (i - 1) * (W_conv - 1) + (minus_j - 1)

                    filter_indices[entry_index] = (0, W_conv - 1, 0, k)
                    filter_values[entry_index] = -1.0
                    entry_index += 1

                    filter_indices[entry_index] = (i, W_conv - 1 + j, 0, k)
                    filter_values[entry_index] = 1.0
                    entry_index += 1

            filter_indices_tf = tf.constant(filter_indices)
            filter_values_tf = tf.cast(tf.constant(filter_values), tf.float64)
            filter_denseshape_tf = tf.constant(filter_denseshape)
            filters_tf_unordered = tf.sparse.SparseTensor(filter_indices_tf, filter_values_tf, filter_denseshape_tf)
            filters_tf = tf.sparse.reorder(filters_tf_unordered)
            filters_tf_dense = tf.sparse.to_dense(filters_tf)
            filters_tf_dense = tf.cast(filters_tf_dense, tf.float64)
            return filters_tf_dense

        @tf.function
        def fix_ensemble_shift(img_convolved: tf.Tensor) -> tf.Tensor:
            """Fixes ensemble shift that results from convolution filter optimization undertaken in make_dense_filters.
            In pseudocode, this does concat(positives, concat(negatives[:upper part down], negatives[:lower part up], axis=h), axis=c).
            """
            lwm1 = W_conv + 1
            lim_area = H_conv * W_conv
            positives = img_convolved[:, :, :, :lim_area]
            negatives = img_convolved[:, :, :, lim_area:]
            negatives_upper = negatives[:, :, lwm1:, :]
            negatives_lower = negatives[:, :, :lwm1, :]
            negatives_fixed = tf.concat([negatives_upper, negatives_lower], 2)
            img_convolved_fixed = tf.concat([positives, negatives_fixed], 3)
            return img_convolved_fixed

        @tf.function
        def normalized_gm(activations: tf.Tensor, layer_area: int, n_channels: int) -> tf.Tensor:
            """Compute normalized Gram matrix."""
            F = tf.reshape(activations, (layer_area, n_channels))
            gram_matrix = tf.linalg.matmul(tf.transpose(F), F)
            normalized_gram_matrix = gram_matrix / layer_area
            return normalized_gram_matrix

        @tf.function
        def l_gram_function(img_thresholded: tf.Tensor) -> tf.Tensor:
            _, img_height, img_width, out_channels = img_thresholded.shape.as_list()
            layer_area = img_height * img_width
            img_gramed = normalized_gm(img_thresholded, layer_area, out_channels)
            return img_gramed

        filters = make_dense_filters()

        @tf.function
        def model(mg_input):
            variogram_list = []
            img_tiled = tile_img(mg_input) if periodic else mg_input
            for bf in range(n_shsh_terms):
                img_tiled_z = tf.expand_dims(img_tiled[:, :, :, bf], axis=-1)
                img_convolved = tf.nn.conv2d(img_tiled_z, filters=filters, strides=[1, 1, 1, 1], padding="VALID")
                img_convolved_fixed = fix_ensemble_shift(img_convolved)
                img_abs = tf.math.abs(img_convolved_fixed)  # Dissertation uses square here, but Alis paper uses abs
                mg_gram = l_gram_function(img_abs) * 0.5
                variogram_list.append(mg_gram)
            return tf.stack(variogram_list, axis=0)

        return model


def register() -> None:
    descriptor_factory.register("OrientationVariogram", OrientationVariogram)
