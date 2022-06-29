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

import pickle
import os
from typing import List

import numpy as np
import tensorflow as tf

from mcrpy.src import descriptor_factory
from mcrpy.descriptors.Descriptor import Descriptor


class GramMatrices(Descriptor):
    is_differentiable = True

    @staticmethod
    def make_singlephase_descriptor(
            desired_shape_2d: tuple = (64, 64),
            nl_method: str = 'relu',
            filename: str = None,
            gram_weights_filename: str = None,
            np_dtype: np.dtype = np.float32,
            tf_dtype: np.dtype = tf.float32,
            periodic: bool = True,
            limit_to: int = 16,
            include_threepoint: bool = True,
            **kwargs):
        """Makes a VGG-19 based differentiable Gram matrix descriptor. 
        The weights are normalized as in
            Gatys et al., Image Style Transfer Using Convolutional Neural 
            Networks, IEEE CVPR, 2016
        Citing this work:
        'We used the feature space provided by a normalised version [...] 
        of the 19-layer VGG network.  We normalized the network by scaling
        the weights such that the mean activation of each convolutional 
        filter over images and positions is equal to one. Such re-scaling 
        can be done for the VGG network without changing its output, because 
        it contains only rectifying linear activation functions and no 
        normalization or pooling over feature maps. We do not use any of 
        the fully connected layers. The model is publicly available [...].  
        [We] found that replacing the maximum pooling operation by average 
        pooling yields slightly more appealing results[.]'
        The same weights are also used in
            Li et al., A Transfer Learning Approach for Microstructure 
            Reconstruction and Structure-property Predictions, Scientific
            Reports, 2018
        and its extension to 3D
            Bostanabad, Reconstruction of 3D Microstructures from 2D Images
            via Transfer Learning, CAD, 2020
        and our 3D work
            Seibert et al., Descriptor-based reconstruction of three-
            dimensional microstructures through gradient-based optimization,
            (submitted), 2021
        The difference of this code with respect to said sources is that here,
        periodic convolutions are used.  Also, phase to RGB encoding is 
        regarded as a part of the descriptor."""
        H, W = desired_shape_2d

        if limit_to < 16:
            raise ValueError(f'limit_to is {limit_to} and should not be less than 16')

        # read weights
        if filename is None:
            if gram_weights_filename is None:
                raise ValueError('Either filename or gram_weights_filename must be given')
            filename = os.path.join(os.path.dirname(__file__), gram_weights_filename)
        with open(filename, 'rb') as f:
            weights_data = pickle.load(f)
        mean_values = weights_data['mean value'].astype(np_dtype)
        weights = weights_data['param values']

        gram_weights = {
            'mg_n1': 0,
            'mg_n2': 1,
            'mg_a1': 0,
            'mg_n3': 0,
            'mg_n4': 1,
            'mg_a2': 0,
            'mg_n5': 0,
            'mg_n6': 0,
            'mg_n7': 0,
            'mg_n8': 1,
            'mg_a3': 0,
            'mg_n9': 0,
            'mg_n10': 0,
            'mg_n11': 0,
            'mg_n12': 1,
            'mg_a4': 0,
            'mg_n13': 0,
            'mg_n14': 0,
            'mg_n15': 0,
            'mg_n16': 1,
            'mg_a5': 0
        }

        @tf.function
        def periodicise_img(img: tf.Tensor, margin: int = 1, symmetric: bool = True) -> tf.Tensor:
            """Add a periodic margin of given width to an image. Needed for periodic convolution boundary conditions."""
            _, h, w, _ = img.shape.as_list()
            margin_multiplier = 2 if symmetric else 1
            h_desired = h + margin_multiplier * margin
            w_desired = w + margin_multiplier * margin

            pre_tiler = np.zeros((h_desired, h), dtype=np_dtype)
            for i in range(h_desired):
                pre_tiler[i % h_desired, (i - margin) % h] = 1
            pre_tiler_tf = tf.constant(pre_tiler)
            post_tiler = np.zeros((w, w_desired), dtype=np_dtype)
            for i in range(w_desired):
                post_tiler[(i - margin) % w, i % w_desired] = 1
            post_tiler_tf = tf.constant(post_tiler)
            img_tiled = tf.einsum(
                'ih,bhxc->bixc', pre_tiler_tf, tf.einsum('bhwc,wx->bhxc', img, post_tiler_tf))
            return img_tiled

        def make_conv2d_layer(weights: tf.Tensor, strides: tf.Tensor = [1, 1, 1, 1]) -> callable:
            """Make a 2D convolution layer with possibility to use periodic boundary conditions, currently not available in std tf."""
            if periodic:
                @tf.function
                def conv2d_layer(ms_pre_conv):
                    ms_periodic = periodicise_img(ms_pre_conv)
                    ms_post_conv = tf.nn.conv2d(
                        ms_periodic, filters=weights, strides=strides, padding='VALID')
                    return ms_post_conv
            else:
                @tf.function
                def conv2d_layer(ms_pre_conv):
                    ms_post_conv = tf.nn.conv2d(
                        ms_pre_conv, filters=weights, strides=strides, padding='SAME')
                    return ms_post_conv
            return conv2d_layer

        def make_encoding_layer() -> callable:
            """Make layer that encodes a microstructure between 0 and 1 to RGB 0..255, then subtract the mean of the training data set."""
            extension_filters = tf.constant(
                np.zeros((1, 1, 1, 3), dtype=np_dtype) + 255)
            subtraction_biases = tf.constant(-mean_values)

            @tf.function
            def encoding_layer(ms_in: tf.Tensor) -> tf.Tensor:
                ms_extended = tf.nn.conv2d(
                    ms_in, extension_filters, strides=(1, 1), padding='VALID')
                ms_shifted = tf.nn.bias_add(
                    ms_extended, subtraction_biases)
                return ms_shifted
            return encoding_layer

        def make_nl_layer(bias: tf.Tensor, nl_method: str = 'relu') -> callable:
            """Make a nonlinearity layer. These nonlinearities are all available by std, this is a simple dev switch. """
            if nl_method == 'relu':
                @tf.function
                def nl_layer(ms_pre_bias):
                    ms_post_bias = tf.nn.relu(ms_pre_bias + bias)
                    return ms_post_bias
            elif nl_method == 'gelu':
                @tf.function
                def nl_layer(ms_pre_bias):
                    ms_post_bias = tf.nn.gelu(ms_pre_bias + bias)
                    return ms_post_bias
            elif nl_method == 'silu':
                @tf.function
                def nl_layer(ms_pre_bias):
                    ms_post_bias = tf.nn.silu(ms_pre_bias + bias)
                    return ms_post_bias
            elif nl_method == 'elu':
                @tf.function
                def nl_layer(ms_pre_bias):
                    ms_post_bias = tf.nn.elu(ms_pre_bias + bias)
                    return ms_post_bias
            elif nl_method == 'leaky_relu':
                @tf.function
                def nl_layer(ms_pre_bias):
                    ms_post_bias = tf.nn.leaky_relu(
                        ms_pre_bias + bias, alpha=0.05)
                    return ms_post_bias
            else:
                raise NotImplementedError
            return nl_layer

        def transpose(weights: np.ndarray) -> np.ndarray:
            """Transpose weights in a special way. Only needed because they are stored in a weird way.  """
            return np.transpose(weights, [2, 3, 1, 0])

        @tf.function
        def compute_normalized_gram_matrix(activations: tf.Tensor, layer_area: int, n_channels: int) -> tf.Tensor:
            """Compute normalized Gram matrix.  """
            F = tf.reshape(activations, (layer_area, n_channels))
            gram_matrix = tf.linalg.matmul(tf.transpose(F), F)
            normalized_gram_matrix = gram_matrix / layer_area
            return normalized_gram_matrix

        def make_avgpool_layer(kernel_size: List[int] = [1, 2, 2, 1], strides: List[int] = [1, 2, 2, 1], padding: str = 'VALID') -> callable:
            """Make pooling layer. These layer are available by std (except blurpool), but this function acts as a simple dev switch.  """
            @tf.function
            def avgpool_layer(ms_pre_avgpool: tf.Tensor) -> tf.Tensor:
                ms_post_avgpool = tf.nn.avg_pool(ms_pre_avgpool, ksize=kernel_size, strides=strides, padding=padding)
                return ms_post_avgpool
            return avgpool_layer

        def make_gram_layer(weight: int = 1) -> callable:
            """Make a layer that computes a gram matrix."""
            @tf.function
            def l_gram(ms_pre_gram: tf.Tensor) -> tf.Tensor:
                _, img_height, img_width, out_channels = ms_pre_gram.shape.as_list()
                layer_area = img_height * img_width
                ms_post_gram = compute_normalized_gram_matrix(
                    ms_pre_gram, layer_area, out_channels)
                return ms_post_gram * weight

            if include_threepoint:
                return l_gram
            else:
                @tf.function
                def l_gram_diag(ms_pre_gram: tf.Tensor) -> tf.Tensor:
                    ms_post_gram = tf.linalg.tensor_diag_part(
                        l_gram(ms_pre_gram))
                    return ms_post_gram
                return l_gram_diag

        mg_input = tf.keras.Input(shape=(H, W, 1), dtype=tf_dtype)
        mg_enc = tf.keras.layers.Lambda(make_encoding_layer())(mg_input)
        mg_c1 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[0])))(mg_enc)
        mg_n1 = tf.keras.layers.Lambda(make_nl_layer(
            weights[1], nl_method=nl_method))(mg_c1)
        mg_c2 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[2])))(mg_n1)
        mg_n2 = tf.keras.layers.Lambda(make_nl_layer(
            weights[3], nl_method=nl_method))(mg_c2)
        mg_a1 = tf.keras.layers.Lambda(make_avgpool_layer())(mg_n2)
        mg_c3 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[4])))(mg_a1)
        mg_n3 = tf.keras.layers.Lambda(make_nl_layer(
            weights[5], nl_method=nl_method))(mg_c3)
        mg_c4 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[6])))(mg_n3)
        mg_n4 = tf.keras.layers.Lambda(make_nl_layer(
            weights[7], nl_method=nl_method))(mg_c4)
        mg_a2 = tf.keras.layers.Lambda(make_avgpool_layer())(mg_n4)
        mg_c5 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[8])))(mg_a2)
        mg_n5 = tf.keras.layers.Lambda(make_nl_layer(
            weights[9], nl_method=nl_method))(mg_c5)
        mg_c6 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[10])))(mg_n5)
        mg_n6 = tf.keras.layers.Lambda(make_nl_layer(
            weights[11], nl_method=nl_method))(mg_c6)
        mg_c7 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[12])))(mg_n6)
        mg_n7 = tf.keras.layers.Lambda(make_nl_layer(
            weights[13], nl_method=nl_method))(mg_c7)
        mg_c8 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[14])))(mg_n7)
        mg_n8 = tf.keras.layers.Lambda(make_nl_layer(
            weights[15], nl_method=nl_method))(mg_c8)
        mg_a3 = tf.keras.layers.Lambda(make_avgpool_layer())(mg_n8)
        mg_c9 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[16])))(mg_a3)
        mg_n9 = tf.keras.layers.Lambda(make_nl_layer(
            weights[17], nl_method=nl_method))(mg_c9)
        mg_c10 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[18])))(mg_n9)
        mg_n10 = tf.keras.layers.Lambda(make_nl_layer(
            weights[19], nl_method=nl_method))(mg_c10)
        mg_c11 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[20])))(mg_n10)
        mg_n11 = tf.keras.layers.Lambda(make_nl_layer(
            weights[21], nl_method=nl_method))(mg_c11)
        mg_c12 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[22])))(mg_n11)
        mg_n12 = tf.keras.layers.Lambda(make_nl_layer(
            weights[23], nl_method=nl_method))(mg_c12)
        mg_a4 = tf.keras.layers.Lambda(make_avgpool_layer())(mg_n12)
        mg_c13 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[24])))(mg_a4)
        mg_n13 = tf.keras.layers.Lambda(make_nl_layer(
            weights[25], nl_method=nl_method))(mg_c13)
        mg_c14 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[26])))(mg_n13)
        mg_n14 = tf.keras.layers.Lambda(make_nl_layer(
            weights[27], nl_method=nl_method))(mg_c14)
        mg_c15 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[28])))(mg_n14)
        mg_n15 = tf.keras.layers.Lambda(make_nl_layer(
            weights[29], nl_method=nl_method))(mg_c15)
        mg_c16 = tf.keras.layers.Lambda(
            make_conv2d_layer(transpose(weights[30])))(mg_n15)
        mg_n16 = tf.keras.layers.Lambda(make_nl_layer(
            weights[31], nl_method=nl_method))(mg_c16)
        mg_a5 = tf.keras.layers.Lambda(make_avgpool_layer())(mg_n16)

        layer_name_to_layer = {
            'mg_n1': mg_n1,
            'mg_n2': mg_n2,
            'mg_a1': mg_a1,
            'mg_n3': mg_n3,
            'mg_n4': mg_n4,
            'mg_a2': mg_a2,
            'mg_n5': mg_n5,
            'mg_n6': mg_n6,
            'mg_n7': mg_n7,
            'mg_n8': mg_n8,
            'mg_a3': mg_a3,
            'mg_n9': mg_n9,
            'mg_n10': mg_n10,
            'mg_n11': mg_n11,
            'mg_n12': mg_n12,
            'mg_a4': mg_a4,
            'mg_n13': mg_n13,
            'mg_n14': mg_n14,
            'mg_n15': mg_n15,
            'mg_n16': mg_n16,
            'mg_a5': mg_a5
        }
        gram_layers = []
        for layer_name, layer_weight in gram_weights.items():
            if layer_weight > 0:
                mg_gram = tf.keras.layers.Lambda(make_gram_layer(
                    weight=layer_weight))(layer_name_to_layer[layer_name])
                n_c1, n_c2 = mg_gram.shape.as_list()
                mg_gram_flattened = tf.reshape(mg_gram, (1, n_c1 * n_c2))
                gram_layers.append(mg_gram_flattened)
        outputs = tf.keras.layers.Concatenate()(
            gram_layers) if len(gram_layers) > 1 else gram_layers[0]
        model = tf.keras.Model(inputs=mg_input, outputs=outputs)
        return model


def register() -> None:
    descriptor_factory.register("GramMatrices", GramMatrices)
