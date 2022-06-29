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

from typing import Any, Callable, Dict, List, Set, Tuple, Union
import logging
import os

from mcrpy.descriptors.Descriptor import Descriptor

import numpy as np
import tensorflow as tf

descriptor_classes: Dict[str, Callable[..., Descriptor]] = {}
descriptor_choices = [d[:-3] for d in os.listdir(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'descriptors')) 
    if d.endswith('.py') and d not in {'Descriptor.py', '__init__.py'}]


def register(
        descriptor_type: str,
        descriptor_class: Descriptor) -> None:
    descriptor_classes[descriptor_type] = descriptor_class


def unregister(descriptor_type: str) -> None:
    descriptor_classes.pop(descriptor_type, None)


def get_visualization(descriptor_type: str) -> None:
    return descriptor_classes[descriptor_type].visualize


def get_class(descriptor_type: str) -> Descriptor:
    try:
        descriptor_class = descriptor_classes[descriptor_type]
    except KeyError:
        raise ValueError(f"Unknown descriptor type {descriptor_type!r}") from None
    return descriptor_class

def create(descriptor_type: str, 
        arguments: Dict[str, Any],
        assert_differentiable: bool = False) -> callable:
    if assert_differentiable:
        if not descriptor_classes[descriptor_type].is_differentiable:
            raise ValueError(f'The {descriptor_type} descriptor is not differentiable.')
    try:
        creator_func = descriptor_classes[descriptor_type].make_descriptor
    except KeyError:
        raise ValueError(f"Unknown descriptor type {descriptor_type!r}") from None
    args_copy = arguments.copy()
    return creator_func(**args_copy)

def permute(
        descriptor_function: Descriptor, 
        shape_3d: Tuple[int],
        np_dtype: np.dtype, 
        tf_dtype: tf.DType, 
        n_phases: int,
        isotropic: bool = False,
        mode: str = 'average') -> Callable:
    assert len(shape_3d) == 3
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

    def get_slice_descriptor(x, spatial_dim, n_slice_outer) -> np.ndarray:
        x_s2b = tf.space_to_batch(x, block_shapes[spatial_dim], paddings)
        x_e_reshaped = tf.reshape(x_s2b[n_slice_outer], batch_element_shapes[spatial_dim])
        slice_descriptor = descriptor_function(x_e_reshaped)
        return slice_descriptor.numpy()

    if mode.lower() == 'average':
        def permutation_loop(x):
            descriptors = []
            for spatial_dim in range(3):
                logging.info(f'permutation loop in dimension {spatial_dim + 1}')
                dim_descriptor = 0
                for n_slice_outer in range(shape_3d[spatial_dim]):
                    logging.info(f'permutation loop {spatial_dim + 1}: {n_slice_outer + 1}')
                    dim_descriptor += get_slice_descriptor(x, spatial_dim, n_slice_outer)
                dim_descriptor /= shape_3d[spatial_dim]
                descriptors.append(dim_descriptor)
            if isotropic:
                return np.sum(descriptors, axis=0) / 3
            return tuple(descriptors)
    elif mode.lower() == 'sample':
        def permutation_loop(x):
            descriptors = []
            for spatial_dim in range(3):
                n_slice_outer = np.random.randint(0, high=shape_3d[spatial_dim])
                descriptors.append(get_slice_descriptor(
                    x, spatial_dim, n_slice_outer))
            if isotropic:
                dim_choice = np.random.randint(0, high=3)
                return descriptors[dim_choice]
            return tuple(descriptors)
    elif mode.lower() == 'sample_surface':
        def permutation_loop(x):
            descriptors = []
            for spatial_dim in range(3):
                descriptors.append(get_slice_descriptor(x, spatial_dim, 0))
            if isotropic:
                dim_choice = np.random.randint(0, high=3)
                return descriptors[dim_choice]
            return tuple(descriptors)
    else:
        raise ValueError(f'Slice mode {mode} not available.')
    return permutation_loop

