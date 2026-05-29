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


class Variogram(PhaseDescriptor):
    """3-point extension of variogram, similar to 3-point correlations"""

    is_differentiable = True

    @staticmethod
    def make_singlephase_descriptor(
        desired_shape_2d=(64, 64),
        limit_to: int = 8,
        l_threshold_value: float = 0.75,
        threshold_steepness: float = 10,
        **kwargs,
    ) -> callable:
        """Makes a function that computes the differentiable two- and three-point multigrid auto-correlation function of phase 1 given a field.
        Differs from make_diff_correlations in that the multigrid version is computed.
        For more information, see the paper:
            Seibert, Ambati, Raßloff, Kästner, Reconstructing random heterogeneous media through differentiable optimization, 2021.
        Note that this function does not perform the increasing soft threshold correction derived in the appendix of the mentioned paper.py .
        It therefore computes \bar{S}, not \tilde{S}. This is fine, but requires D^\text{des} to be computed accordingly.
        """
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

        @tf.function
        def make_dense_filters() -> tf.Tensor:
            """Make the convolution filter masks for the convolve threshold reduce pipeline for computing spatial correlations in a differentiable manner.
            This includes negative correlation vector components.
            The symmetry of the ensemble with respect to reversing the direction of the correlation vector is exploited by computing only half the correlations.
            As a further code optimization, to reduce the stencil size, those masks that correspond to negative indices are shifted such that they lie
            entirely in the positive index quadrant, with the correlation vector not starting at the origin but above.
            This shifts the resulting ensembles up, making them equally useable for computing two-point correlations, but not usable for three-point correlations.
            Therefore, they are shifted back again later in a different substep via the function fix_ensemble_shift.
            Furthermore, note that the masks are constructed in a sparse manner that would enable the usage of a sparse array data type that can
            exploit the extreme sparsity of these masks.
            However, at the moment of writing this, TensorFlow does not support convolutions with sparse kernels, which are admittedly hard to optimize.
            Therefore, the sparse data structure is converted to dense at the end of the function."""
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
                        filter_values[entry_index] = 0
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
            img_tiled = tile_img(mg_input)
            img_convolved = tf.nn.conv2d(img_tiled, filters=filters, strides=[1, 1, 1, 1], padding="VALID")
            img_convolved_fixed = fix_ensemble_shift(img_convolved)
            img_squared = tf.math.abs(img_convolved_fixed)
            # img_squared = tf.math.square(img_convolved_fixed)
            mg_gram = l_gram_function(img_squared) * 0.5
            mg_gram_diag = tf.linalg.tensor_diag_part(mg_gram)
            return mg_gram_diag

        return model

    @classmethod
    def visualize_subplot(
        cls,
        descriptor_value: np.ndarray,
        ax,
        axis: bool = True,
        descriptor_type: str = None,
        mg_level: int = None,
        n_phase: int = None,
    ):
        x_max = descriptor_value.shape
        assert len(x_max) == 1
        x_max = x_max[0]
        limit_to = np.round(0.5 + np.sqrt(0.5 * x_max - 0.25), decimals=0).astype(int)
        xticks = [0, limit_to - 1, 2 * (limit_to - 1)]
        yticks = [0, limit_to - 1, 2 * (limit_to - 1)]
        s2_descriptor = descriptor_value.flatten()
        s2_sorted = np.zeros(tuple([2 * limit_to - 1] * 2))
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
        ax.imshow(s2_sorted, cmap="cividis")
        ax.set_title(f"var: l={mg_level}, p={n_phase}")
        ax.set_xlabel(r"$r_x$ in Px")
        ax.set_ylabel(r"$r_y$ in Px")
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([-limit_to + 1, 0, limit_to - 1])
        ax.set_yticklabels(reversed([-limit_to + 1, 0, limit_to - 1]))


def register() -> None:
    descriptor_factory.register("Variogram", Variogram)
