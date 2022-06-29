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

import tensorflow as tf
import numpy as np

def make_2d_gradients(loss_function):
    def gradient_computation(x):
        optimize_var = [x]
        with tf.GradientTape(persistent=False) as tape:
            loss = loss_function(x)
            grads = tape.gradient(loss, optimize_var)
        return loss, grads
    return gradient_computation

def make_3d_gradients(loss_function, shape_3d, n_phases, np_dtype, stride=64):
    """Low-memory version of permutation_operator_wrap that replaces not f_C, but L directly. Under construction.

    Args:
        loss_function (tf.function, tuple): Loss function to wrap. Tuple of loss functions in anisotropic case.
        shape_3d (tuple): 3D shape of microstructure to reconstruct.
        np_dtype (np.dtype): Numpy data type for flexible precision.
        stride (int, optional): Stride for unrolling and jitting loop. Optional. Defaults to 4.
    """
    stride = min(stride, min(shape_3d))
    anisotropic = isinstance(loss_function, tuple) or isinstance(loss_function, list)

    dtype_dict = {np.float16: tf.float16, np.float32: tf.float32, np.float64: tf.float64}
    tf_dtype = dtype_dict[np_dtype]
    zero_grads = tf.constant(np.zeros((1, *shape_3d, n_phases), dtype=np_dtype), dtype=tf_dtype)
    grads = tf.Variable(initial_value=zero_grads, trainable=False, dtype=tf_dtype)
    inner_grads = tf.Variable(initial_value=zero_grads, trainable=False, dtype=tf_dtype)

    paddings = tf.constant(np.zeros((4, 2), dtype=np.int32), dtype=tf.int32)
    block_shapes_np = np.zeros((3, 4), dtype=np.int32)
    for n_dim, dim_shape in enumerate(shape_3d):
        block_shape = np.ones(4)
        block_shape[n_dim] = dim_shape
        block_shapes_np[n_dim] = block_shape
    block_shapes = tf.constant(block_shapes_np, dtype=tf.int32)
    shape_per_batch_element = tf.constant((*shape_3d, n_phases))
    batch_element_shapes = [
            (1, shape_3d[1], shape_3d[2], n_phases),
            (1, shape_3d[0], shape_3d[2], n_phases),
            (1, shape_3d[0], shape_3d[1], n_phases),
            ]

    @tf.function
    def partially_unrolled_loop(x, spatial_dim, n_slice_outer, use_loss, batch_element_shape):
        inner_grads.assign(zero_grads)
        inner_loss = 0
        optimize_var = [x]
        for add_offset in range(stride):
            n_slice = n_slice_outer + add_offset
            with tf.GradientTape() as tape:
                x_s2b = tf.space_to_batch(x, block_shapes[spatial_dim], paddings)
                x_e_reshaped = tf.reshape(x_s2b[n_slice], batch_element_shape)
                x_e_reshaped.set_shape(batch_element_shape)
                inner_loss = inner_loss + use_loss(x_e_reshaped)
                inner_grads.assign_add(tape.gradient(inner_loss, optimize_var)[0])
        return inner_loss, inner_grads

    def repetitive_gradient_accumulation(x):
        grads.assign(zero_grads)
        total_loss = 0
        for spatial_dim in range(3):
            use_loss = loss_function[spatial_dim] if anisotropic else loss_function
            for n_slice_outer in range(0, shape_3d[spatial_dim], stride):
                partial_loss, partial_grads = partially_unrolled_loop(x, tf.constant(spatial_dim), tf.constant(n_slice_outer), use_loss, batch_element_shapes[spatial_dim])
                total_loss = total_loss + partial_loss
                grads.assign_add(partial_grads)
        return total_loss, [grads]
    return repetitive_gradient_accumulation

def make_3d_nograds(loss_function, shape_3d, n_phases):
    anisotropic = isinstance(loss_function, tuple) or isinstance(loss_function, list)
    paddings = tf.constant(np.zeros((4, 2), dtype=np.int32), dtype=tf.int32)
    block_shapes_np = np.zeros((3, 4), dtype=np.int32)
    for n_dim, dim_shape in enumerate(shape_3d):
        block_shape = np.ones(4)
        block_shape[n_dim] = dim_shape
        block_shapes_np[n_dim] = block_shape
    block_shapes = tf.constant(block_shapes_np, dtype=tf.int32)
    batch_element_shapes = [
            (1, shape_3d[1], shape_3d[2], n_phases),
            (1, shape_3d[0], shape_3d[2], n_phases),
            (1, shape_3d[0], shape_3d[1], n_phases),
            ]

    def loss_accumulation(x):
        total_loss = 0
        for spatial_dim in range(3):
            use_loss = loss_function[spatial_dim] if anisotropic else loss_function
            for n_slice in range(shape_3d[spatial_dim]):
                x_s2b = tf.space_to_batch(x, block_shapes[spatial_dim], paddings)
                x_e_reshaped = tf.reshape(x_s2b[n_slice], batch_element_shapes[spatial_dim])
                x_e_reshaped.set_shape(batch_element_shapes[spatial_dim])
                total_loss = total_loss + use_loss(x_e_reshaped)
        return total_loss
    return loss_accumulation

def make_call_loss(loss_function, desired_shape_ms, n_phases, np_dtype, is_gradient_based, stride: int = 64):
    ms_is_2d = len(desired_shape_ms) == 2
    ms_is_3d = len(desired_shape_ms) == 3
    if ms_is_2d and not is_gradient_based:
        return loss_function
    if ms_is_2d and is_gradient_based:
        return make_2d_gradients(loss_function)
    elif ms_is_3d and not is_gradient_based:
        return make_3d_nograds(loss_function, desired_shape_ms, n_phases)
    elif ms_is_3d and is_gradient_based:
        return make_3d_gradients(loss_function, desired_shape_ms, n_phases, np_dtype, stride=stride)
    else:
        raise ValueError('Desired shape must be 2D or 3D')
