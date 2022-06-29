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
from mcrpy.descriptors.Descriptor import Descriptor


class TwoPointCorrelations(Descriptor):
    is_differentiable = True

    @staticmethod
    def make_singlephase_descriptor( 
            desired_shape_2d=(64, 64), 
            limit_to: int = 8, 
            tf_dtype=tf.float64, 
            l_threshold_value=0.75, 
            threshold_steepness=10, 
            **kwargs) -> callable:
        H, W = desired_shape_2d
        H_conv = limit_to
        W_conv = limit_to
        z_lower_bound = tf.cast(1.0 / (1.0 + tf.math.exp(-((0.0 - l_threshold_value) * threshold_steepness))), dtype=tf_dtype)
        z_upper_bound = tf.cast(1.0 / (1.0 + tf.math.exp(-((1.0 - l_threshold_value) * threshold_steepness))), dtype=tf_dtype)
        a = tf.cast(1.0 / (z_upper_bound - z_lower_bound), dtype=tf_dtype)
        b = tf.cast(- a * z_lower_bound, dtype=tf_dtype)


        @tf.function
        def tile_img(img: tf.Tensor) -> tf.Tensor:
            """Tile an image. Needed for periodic boundary conditions in convolution."""
            _, h, w, _ = img.shape.as_list()
            img_shape_2d = (h, w)
            lim_height = min(H_conv, h)
            lim_width = min(W_conv, w)
            r_max = (lim_height, lim_width)

            tiled_shape = np.array(img_shape_2d) + np.array(r_max)
            h_desired = h + lim_height
            w_desired = w + lim_width

            pre_tiler = np.zeros((h_desired, h), dtype=np.int8)
            for i in range(h_desired):
                pre_tiler[i % h_desired, i % h] = 1
            pre_tiler_tf = tf.cast(tf.constant(pre_tiler), tf_dtype)
            post_tiler = np.zeros((w, w_desired), dtype=np.int8)
            for i in range(w_desired):
                post_tiler[i % w, i % w_desired] = 1
            post_tiler_tf = tf.cast(tf.constant(post_tiler), tf_dtype)
            img_tiled = tf.reshape(tf.linalg.matmul(pre_tiler_tf, tf.linalg.matmul(tf.reshape(img, img_shape_2d), post_tiler_tf)), [1] + [*tiled_shape] + [1])
            return img_tiled


        @tf.function
        def make_dense_filters() -> tf.Tensor:
            in_channels = 1
            out_channels = H_conv * W_conv + (H_conv - 1) * (W_conv - 1)
            n_entries = out_channels * 2 - 1
            filter_indices = np.zeros((n_entries, 4), dtype=np.int64)
            filter_values = np.zeros((n_entries), dtype=np.float64)
            filter_denseshape = np.array([H_conv + 1, W_conv + 1, in_channels, out_channels], dtype=np.int64)
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
                    filter_values[entry_index] = 0.5
                    entry_index += 1

                    filter_indices[entry_index] = (i, j, 0, k)
                    filter_values[entry_index] = 0.5
                    entry_index += 1
            for i in range(1, H_conv):
                for minus_j in range(1, W_conv):
                    j = -minus_j
                    k = H_conv * W_conv + (i - 1) * (W_conv - 1) + (minus_j - 1)

                    filter_indices[entry_index] = (0, W_conv - 1, 0, k)
                    filter_values[entry_index] = 0.5
                    entry_index += 1

                    filter_indices[entry_index] = (i, W_conv - 1 + j, 0, k)
                    filter_values[entry_index] = 0.5
                    entry_index += 1

            filter_indices_tf = tf.constant(filter_indices)
            filter_values_tf = tf.cast(tf.constant(filter_values), tf_dtype)
            filter_denseshape_tf = tf.constant(filter_denseshape)
            filters_tf_unordered = tf.sparse.SparseTensor(filter_indices_tf, filter_values_tf, filter_denseshape_tf)
            filters_tf = tf.sparse.reorder(filters_tf_unordered) 
            filters_tf_dense = tf.sparse.to_dense(filters_tf)
            filters_tf_dense = tf.cast(filters_tf_dense, tf_dtype)
            return filters_tf_dense

        @tf.function
        def fix_ensemble_shift(img_convolved: tf.Tensor) -> tf.Tensor:
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
            F = tf.reshape(activations, (layer_area, n_channels))
            gram_matrix = tf.linalg.matmul(tf.transpose(F), F)
            normalized_gram_matrix = gram_matrix / layer_area
            return normalized_gram_matrix

        @tf.function
        def l_gram(img_thresholded: tf.Tensor) -> tf.Tensor:
            _, img_height, img_width, out_channels = img_thresholded.shape.as_list()
            layer_area = img_height * img_width
            img_gramed = normalized_gm(img_thresholded, layer_area, out_channels)
            return img_gramed
        @tf.function
        def l_gram_function(img_thresholded: tf.Tensor) -> tf.Tensor:
            img_gramed = tf.linalg.tensor_diag_part(l_gram(img_thresholded))
            return img_gramed
        filters = make_dense_filters()

        @tf.function
        def model(mg_input):
            img_tiled = tile_img(mg_input)
            img_convolved = tf.nn.conv2d(img_tiled, filters=filters,
                    strides=[1, 1, 1, 1], padding='VALID')
            img_convolved_fixed = fix_ensemble_shift(img_convolved)
            img_thresholded = tf.nn.sigmoid((img_convolved_fixed - l_threshold_value) * threshold_steepness) * a + b
            mg_gram = l_gram_function(img_thresholded)
            return mg_gram
        return model

    @staticmethod
    def define_comparison_mask(
            desired_descriptor_shape: Tuple[int] = None, 
            limit_to: int = None, 
            **kwargs):
        assert len(desired_descriptor_shape) == 1
        desired_limit_to = np.round(0.5 + np.sqrt(0.5 * desired_descriptor_shape[0] - 0.25),
                decimals=0).astype(int)
        logging.info(f'limit_to for desired_descriptor is {desired_limit_to}')
        logging.info(f'limit_to for current is {limit_to}')

        if limit_to == desired_limit_to:
            return None, False

        larger_limit_to = max(limit_to, desired_limit_to)
        smaller_limit_to = min(limit_to, desired_limit_to)
        larger_n_elements = larger_limit_to**2 + (larger_limit_to - 1)**2
        boolean_list = []
        for i in range(larger_limit_to):
            for j in range(larger_limit_to):
                boolean_list.append(i < smaller_limit_to and j < smaller_limit_to)
        for i in range(1, larger_limit_to):
            for j in range(1, larger_limit_to):
                boolean_list.append(i < smaller_limit_to and j < smaller_limit_to)
        mask = np.array(boolean_list, dtype=np.bool8)
        return mask, limit_to > desired_limit_to

    @classmethod
    def visualize_subplot(
            cls,
            descriptor_value: np.ndarray,
            ax,
            descriptor_type: str = None,
            mg_level: int = None,
            n_phase: int = None):
        x_max = descriptor_value.shape
        limit_to = np.round(0.5 + np.sqrt(0.5 * x_max - 0.25), decimals=0).astype(int)
        xticks = [0, limit_to - 1, 2 * (limit_to - 1)]
        yticks = [0, limit_to - 1, 2 * (limit_to - 1)]
        s2_descriptor = descriptor_value
        s2_sorted = np.zeros(tuple([2 * limit_to - 1]*2))
        k = 0
        for i in range(limit_to):
            for j in range(limit_to):
                s2_sorted[limit_to - 1 + i, limit_to - 1 + j] = s2_descriptor[k]
                s2_sorted[limit_to - 1 - i, limit_to - 1 - j] = s2_descriptor[k]
                k += 1
        for i in range(1, limit_to):
            for j in range(1, limit_to):
                s2_sorted[limit_to - 1 + i, limit_to - 1 - j] = s2_descriptor[k]
                s2_sorted[limit_to - 1 - i, limit_to - 1 + j] = s2_descriptor[k]
                k += 1
        ax.imshow(s2_sorted, cmap='cividis')
        ax.set_title(f'S2: l={mg_level}, p={n_phase}')
        ax.set_xlabel(r'$r_x$ in Px')
        ax.set_ylabel(r'$r_y$ in Px')
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([-limit_to + 1, 0, limit_to - 1])
        ax.set_yticklabels(reversed([-limit_to + 1, 0, limit_to - 1]))


def register() -> None:
    descriptor_factory.register("TwoPointCorrelations", TwoPointCorrelations)
