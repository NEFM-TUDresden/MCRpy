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
from typing import Union, Tuple, List

import tensorflow as tf
import numpy as np

from mcrpy.src.Microstructure import Microstructure

def make_2d_nograds(loss_function: callable) -> callable:
    """Return a function that computes the loss function given a 2D 
    microstructure. 
    """
    def loss_computation_inner(ms: Microstructure):
        loss = loss_function(ms.xx)
        return loss
    return loss_computation_inner

def make_2d_gradients(loss_function: callable) -> callable:
    """Return a function that computes the loss function and its gradient
    given a 2D microstructure. 
    """
    def gradient_computation(ms: Microstructure):
        optimize_var = [ms.x]
        with tf.GradientTape(persistent=False) as tape:
            loss = loss_function(ms.xx)
            grads = tape.gradient(loss, optimize_var)
        return loss, grads
    return gradient_computation

# def make_2d_gradients(loss_function: callable) -> callable:
#     from jax import value_and_grad
#     from jax.experimental.jax2tf import call_tf, convert
#     def loss_function_typecast(input):
#         return tf.cast(loss_function(tf.cast(input, tf.float64)), tf.float32)
#     lf_jax = call_tf(loss_function_typecast)
#     lg_jax = value_and_grad(lf_jax)
#     lg_tf = convert(lg_jax)
#     def inner(ms: Microstructure):
#         ms_xx = tf.cast(ms.xx, tf.float32)
#         val, grad = lg_tf(ms_xx)
#         return tf.cast(val, tf.float64), tf.cast(grad, tf.float64)
#     return inner

def make_3d_gradients(
        loss_function: Union[callable, List[callable], Tuple[callable]], 
        shape_3d: Tuple[int]) -> callable:
    """Return a function that computes the loss function and its gradient
    given a 3D microstructure. Does not use tf.GradientTape trivially
    because of memory issues, but swaps the gradient and the sum over all
    slices in order to copmute the gradient as a sum of gradients of slice-
    wise loss functions.
    """
    anisotropic = isinstance(loss_function, (tuple, list))

    zero_grads = tf.constant(np.zeros((1, *shape_3d), dtype=np.float64), dtype=tf.float64)
    grads = tf.Variable(initial_value=zero_grads, trainable=False, dtype=tf.float64)
    inner_grads = tf.Variable(initial_value=zero_grads, trainable=False, dtype=tf.float64)

    def repetitive_gradient_accumulation(ms: Microstructure):
        optimize_var = [ms.x]
        grads.assign(zero_grads)
        total_loss = 0
        for spatial_dim in range(3):
            use_loss = loss_function[spatial_dim] if anisotropic else loss_function
            for slice_index in range(ms.spatial_shape[spatial_dim]):
                inner_loss = 0
                with tf.GradientTape() as tape:
                    ms_slice = ms.get_slice(spatial_dim, slice_index)
                    inner_loss = use_loss(ms_slice)
                    partial_grads = tape.gradient(inner_loss, optimize_var)[0]
                total_loss = total_loss + inner_loss
                grads.assign_add(partial_grads)
        return total_loss, [grads]

    return repetitive_gradient_accumulation

def make_3d_gradients_greedy(
        loss_function: Union[callable, List[callable], Tuple[callable]], 
        shape_3d: Tuple[int],
        batch_size: int = 1) -> callable:
    """Greedily compute just a random slice and return the result. """
    anisotropic = isinstance(loss_function, (tuple, list))

    zero_grads = tf.constant(np.zeros((1, *shape_3d), dtype=np.float64), dtype=tf.float64)
    grads = tf.Variable(initial_value=zero_grads, trainable=False, dtype=tf.float64)

    def repetitive_gradient_accumulation(ms: Microstructure):
        optimize_var = [ms.x]
        grads.assign(zero_grads)
        total_loss = 0
        for spatial_dim in range(3):
            use_loss = loss_function[spatial_dim] if anisotropic else loss_function
            n_slices = (
                batch_size
                if batch_size >= 1
                else int(ms.spatial_shape[spatial_dim] * batch_size)
            )
            slice_indices = tf.cast(tf.random.uniform(
                [n_slices], minval=0, maxval=ms.spatial_shape[spatial_dim]),
                tf.int32)
            for n_slice in range(n_slices):
                with tf.GradientTape() as tape:
                    ms_slice = ms.get_slice(spatial_dim, slice_indices[n_slice])
                    inner_loss = use_loss(ms_slice)
                    partial_grads = tape.gradient(inner_loss, optimize_var)[0]
                total_loss = total_loss + inner_loss
                grads.assign_add(partial_grads)
        return total_loss, [grads]

    return repetitive_gradient_accumulation

def make_3d_star(loss_function: callable) -> callable:
    """Return a function that computes the loss function and its gradient
    given a 3D microstructure for a sparse loss function. Unlike, the function
    make_3d_gradients, which computes the loss and its gradients on all slices
    of the Microstructure, this function only considers 6 slices: Assuming that
    only two pixels have changed in the Microstructure, for example, during an
    iteration of the Yeong-Torquato algorithm, it is only required to probe
    these 6 slices to compute whether the pixel swap was good or not. While this
    greatly improves efficiency ofer a dense implementation, it also means that
    the loss values of the Yeong-Torquato algorithm in two subsequent iterations
    are completely unrelated and cannot be interpreted well. The indices of the
    swapped pixels are not given, but are expected to be store in the
    Microstructure object.
    """
    anisotropic = isinstance(loss_function, (tuple, list))

    def loss_accumulation(ms: Microstructure):
        total_loss = 0
        for spatial_dim in range(3):
            use_loss = loss_function[spatial_dim] if anisotropic else loss_function
            for indices in [ms.swapped_index_1, ms.swapped_index_2]:
                n_slice = indices[spatial_dim]
                ms_slice = ms.get_slice(spatial_dim, n_slice)
                total_loss = total_loss + use_loss(ms_slice)
        return total_loss

    return loss_accumulation


def make_3d_nograds(
        loss_function: Union[callable, List[callable], Tuple[callable]], 
        shape_3d: Tuple[int]) -> callable:
    """Same as make_3d_star, but without gradients."""
    anisotropic = isinstance(loss_function, (tuple, list))

    def loss_accumulation(ms: Microstructure):
        total_loss = 0
        for spatial_dim in range(3):
            use_loss = loss_function[spatial_dim] if anisotropic else loss_function
            for ms_slice in ms.get_slice_iterator(spatial_dim):
                total_loss = total_loss + use_loss(ms_slice)
        return total_loss

    return loss_accumulation

def make_call_loss(
        loss_function: Union[callable, List[callable], Tuple[callable]], 
        ms: Microstructure, 
        is_gradient_based: bool, 
        sparse: bool = False,
        greedy: bool = False,
        batch_size: int = 1) -> callable:  # sourcery skip: remove-redundant-if
    """Make and return a function that computes the loss_function and
    possibly the gradient given a Microstructure.
    """
    ms_is_3d = ms.is_3D
    ms_is_2d = not ms_is_3d
    if ms_is_2d and not is_gradient_based:
        return make_2d_nograds(loss_function)
    if ms_is_2d and is_gradient_based:
        return make_2d_gradients(loss_function)
    elif ms_is_3d and not is_gradient_based:
        if sparse:
            return make_3d_star(loss_function)
        else:
            return make_3d_nograds(loss_function, ms.shape)
    elif ms_is_3d and is_gradient_based:
        if greedy:
            return make_3d_gradients_greedy(loss_function, ms.shape, batch_size=batch_size)
        else:
            return make_3d_gradients(loss_function, ms.shape)
    else:
        raise ValueError('Desired shape must be 2D or 3D')
