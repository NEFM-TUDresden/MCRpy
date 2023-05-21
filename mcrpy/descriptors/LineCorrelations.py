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
from mcrpy.descriptors.PhaseDescriptor import PhaseDescriptor


class LineCorrelations(PhaseDescriptor):
    is_differentiable = True

    @staticmethod
    def make_singlephase_descriptor( 
            desired_shape_2d=(64, 64), 
            limit_to: int = 8, 
            l_threshold_value: float = 0.75, 
            threshold_steepness: float = 10, 
            **kwargs) -> callable:
        """Makes a function that computes the differentiable two- and three-point multigrid auto-correlation function of phase 1 given a field.  
        Differs from make_diff_correlations in that the multigrid version is computed. 
        For more information, see the paper: 
            Seibert, Ambati, Raßloff, Kästner, Reconstructing random heterogeneous media through differentiable optimization, 2021. 
        Note that this function does not perform the increasing soft threshold correction derived in the appendix of the mentioned paper.py .
        It therefore computes \bar{S}, not \tilde{S}. This is fine, but requires D^\text{des} to be computed accordingly."""
        H, W = desired_shape_2d
        H_conv = limit_to
        W_conv = limit_to
        z_lower_bound = tf.cast(1.0 / (1.0 + tf.math.exp(-((0.0 - l_threshold_value) * threshold_steepness))), dtype=tf.float64)
        z_upper_bound = tf.cast(1.0 / (1.0 + tf.math.exp(-((1.0 - l_threshold_value) * threshold_steepness))), dtype=tf.float64)
        a = tf.cast(1.0 / (z_upper_bound - z_lower_bound), dtype=tf.float64)
        b = tf.cast(- a * z_lower_bound, dtype=tf.float64)

        @tf.function
        def make_dense_filters_heightm1() -> tf.Tensor:
            in_channels = 1
            out_channels = H_conv - 1
            n_entries = out_channels * 2
            filter_indices = np.zeros((n_entries, 4), dtype=np.int64)
            filter_values = np.zeros((n_entries), dtype=np.float64)
            filter_denseshape = np.array([H_conv, 1, in_channels, out_channels], dtype=np.int64)
            entry_index = 0
            j = 0
            for i in range(1, H_conv):
                k = i - 1
                if i == 0:
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
            filter_indices_tf = tf.constant(filter_indices)
            filter_values_tf = tf.cast(tf.constant(filter_values), tf.float64)
            filter_denseshape_tf = tf.constant(filter_denseshape)
            filters_tf_unordered = tf.sparse.SparseTensor(filter_indices_tf, filter_values_tf, filter_denseshape_tf)
            filters_tf = tf.sparse.reorder(filters_tf_unordered) 
            filters_tf_dense = tf.sparse.to_dense(filters_tf)
            filters_tf_dense = tf.cast(filters_tf_dense, tf.float64)
            return filters_tf_dense

        @tf.function
        def make_dense_filters_width() -> tf.Tensor:
            in_channels = 1
            out_channels = W_conv
            n_entries = out_channels * 2 - 1
            filter_indices = np.zeros((n_entries, 4), dtype=np.int64)
            filter_values = np.zeros((n_entries), dtype=np.float64)
            filter_denseshape = np.array([1, W_conv, in_channels, out_channels], dtype=np.int64)
            entry_index = 0
            i = 0
            for j in range(W_conv):
                k = j
                if j == 0:
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
            filter_indices_tf = tf.constant(filter_indices)
            filter_values_tf = tf.cast(tf.constant(filter_values), tf.float64)
            filter_denseshape_tf = tf.constant(filter_denseshape)
            filters_tf_unordered = tf.sparse.SparseTensor(filter_indices_tf, filter_values_tf, filter_denseshape_tf)
            filters_tf = tf.sparse.reorder(filters_tf_unordered) 
            filters_tf_dense = tf.sparse.to_dense(filters_tf)
            filters_tf_dense = tf.cast(filters_tf_dense, tf.float64)
            return filters_tf_dense

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

        # filters = make_dense_filters()
        filters_w = make_dense_filters_width()
        filters_h = make_dense_filters_heightm1()
        tile_img_x = make_image_padder(min(W_conv, W) - 1, 0)
        tile_img_y = make_image_padder(0, min(H_conv, H) - 1)


        @tf.function
        def model(mg_input):
            img_tiled_h = tile_img_x(mg_input)
            img_convolved_h = tf.nn.conv2d(img_tiled_h, filters=filters_h, strides=[1, 1, 1, 1], padding='VALID')
            img_tiled_w = tile_img_y(mg_input)
            img_convolved_w = tf.nn.conv2d(img_tiled_w, filters=filters_w, strides=[1, 1, 1, 1], padding='VALID')
            img_convolved = tf.concat([img_convolved_h, img_convolved_w], axis=-1)
            img_thresholded = tf.nn.sigmoid((img_convolved - l_threshold_value) * threshold_steepness) * a + b
            mg_gram = l_gram_function(img_thresholded)
            return mg_gram
        return model

def register() -> None:
    descriptor_factory.register("LineCorrelations", LineCorrelations)
