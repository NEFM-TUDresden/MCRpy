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

from abc import ABC, abstractmethod
import logging

import numpy as np
import tensorflow as tf

class Descriptor(ABC):
    is_differentiable = True

    @classmethod
    def make_descriptor(
            cls, 
            desired_shape_2d=None, 
            desired_shape_extended=None, 
            use_multigrid_descriptor=True, 
            use_multiphase=True, 
            limit_to = 8,
            tf_dtype = None,
            **kwargs) -> callable:
        """By default wraps self.make_single_phase_descriptor."""
        if use_multigrid_descriptor:
            singlephase_descriptor =  cls.make_multigrid_descriptor(
                limit_to=limit_to,
                desired_shape_2d=desired_shape_2d,
                desired_shape_extended=desired_shape_extended,
                tf_dtype=tf_dtype,
                **kwargs)
        else:
            singlephase_descriptor = cls.make_singlegrid_descriptor(
                limit_to=limit_to, 
                desired_shape_2d=desired_shape_2d,
                desired_shape_extended=desired_shape_extended,
                tf_dtype=tf_dtype,
                **kwargs) 
        ms_shape = desired_shape_extended
        n_phases = ms_shape[-1]
        n_pixels = np.prod(ms_shape[:-1])

        @tf.function
        def singlephase_wrapper(x: tf.Tensor) -> tf.Tensor:
            phase_descriptor = tf.expand_dims(singlephase_descriptor(x), axis=0)
            return phase_descriptor

        @tf.function
        def multiphase_wrapper(x: tf.Tensor) -> tf.Tensor:
            phase_descriptors = []
            for phase in range(n_phases):
                x_phase = x[:, :, :, phase]
                phase_descriptor = tf.expand_dims(singlephase_descriptor(tf.expand_dims(x_phase, axis=-1)), axis=0)
                phase_descriptors.append(phase_descriptor)
            return tf.concat(phase_descriptors, axis=0)
        return multiphase_wrapper if use_multiphase else singlephase_wrapper

    @classmethod
    def make_singlegrid_descriptor(
            cls,
            desired_shape_2d=None,
            desired_shape_extended=None,
            limit_to=8,
            tf_dtype=None,
            **kwargs):
        singlephase_kwargs = {
                'desired_shape_2d': desired_shape_2d,
                'desired_shape_extended': desired_shape_extended,
                'tf_dtype': tf_dtype,
                'limit_to': limit_to,
                **kwargs
                }
        unexpanded_singlephase_descriptor = cls.wrap_singlephase_descriptor(**singlephase_kwargs)

        @tf.function
        def singlegrid_descriptor(inputs):
            interm = unexpanded_singlephase_descriptor(inputs)
            outputs = tf.expand_dims(interm, axis=0)
            return outputs
        return singlegrid_descriptor


    @classmethod
    def make_multigrid_descriptor(
            cls,
            desired_shape_2d=None,
            desired_shape_extended=None,
            limit_to=8,
            tf_dtype=None,
            **kwargs):

        H, W = desired_shape_2d
        n_phases = desired_shape_extended[-1]
        limitation_factor = min(H / limit_to, W / limit_to)
        mg_levels = int(np.floor(np.log(limitation_factor) / np.log(2)))
        singlephase_descriptors = []
        for mg_level in range(mg_levels):
            pool_size = 2**mg_level
            if H % pool_size != 0 or W % pool_size != 0:
                logging.warning('For MG level number {}, an avgpooling remainder exists.'.format(mg_level))
            desired_shape_layer = tuple([s//pool_size for s in desired_shape_2d])
            singlephase_kwargs = {
                    'desired_shape_2d': desired_shape_layer,
                    'desired_shape_extended': (1, *desired_shape_layer, n_phases),
                    'tf_dtype': tf_dtype,
                    'limit_to': limit_to,
                    **kwargs
                    }
            singlephase_descriptors.append(cls.wrap_singlephase_descriptor(**singlephase_kwargs))
        else:
            logging.info('Could create all {} MG levels'.format(mg_levels))

        @tf.function
        def multigrid_descriptor(mg_input):
            mg_layers = []
            for mg_level, singlephase_descriptor in enumerate(singlephase_descriptors):
                pool_size = 2**mg_level
                mg_pool = tf.nn.avg_pool2d(mg_input, [pool_size, pool_size], [pool_size, pool_size], 'VALID')
                mg_desc = singlephase_descriptor(mg_pool)
                mg_exp = tf.expand_dims(mg_desc, axis=0)
                mg_layers.append(mg_exp)
            outputs = tf.concat(mg_layers, axis=0) if len(mg_layers) > 1 else mg_layers[0]
            return outputs
        return multigrid_descriptor

    @classmethod
    def wrap_singlephase_descriptor(
            cls, 
            np_dtype: np.dtype = None,
            tf_dtype: tf.DType = None,
            **kwargs):
        if cls.is_differentiable:
            return cls.make_singlephase_descriptor(
                    np_dtype=np_dtype, tf_dtype=tf_dtype, **kwargs)
        else:
            compute_descriptor_np = cls.make_singlephase_descriptor(
                    np_dtype=np_dtype, tf_dtype=tf_dtype, **kwargs)

            def compute_descriptor_tf(x: tf.Tensor) -> tf.Tensor:
                x_np = x.numpy()
                y_np = compute_descriptor_np(x_np)
                y_tf = tf.constant(y_np.astype(np_dtype), dtype=tf_dtype)
                return y_tf

            @tf.function 
            def compute_descriptor_compiled(x: tf.Tensor) -> tf.Tensor:
                py_descriptor = tf.py_function(func=compute_descriptor_tf, inp=[x], Tout=tf_dtype)
                return py_descriptor
            return compute_descriptor_compiled

    @classmethod
    def make_singlephase_descriptor(
            cls, 
            np_dtype: np.dtype = None,
            tf_dtype: tf.DType = None,
            **kwargs) -> callable:
        """Staticmethod that return a function that computes the descriptor of a single phase MS.
        For differentiable descriptors (cls.is_differentiable), this function should take the MS
        as a tensorflow variable and return the descriptor as a tensorflow variable and should be
        differentiable. For non-differentiable descriptors (not cls.is_differentiable), this
        function should take the MS as a np.ndarray and return the descriptor as a np.ndarray.
        It will be wrapped automatically using tf.py_function."""
        raise NotImplementedError("Implement this in all used Descriptor subclasses")

    @staticmethod
    def define_comparison_mask(
            desired_descriptor_shape: Tuple[int] = None, 
            limit_to: int = None, 
            **kwargs):
        """Defines a mask for the case that two descriptors need to be compared and the shape
        doesn't match. As an example, see FFTCorrelations.py. The second return value determines
        if the current descriptor shape is larger than the desired (True) or not (False)."""
        return None, False

    @classmethod
    def make_comparison(cls, **kwargs):
        mask, swap_args = cls.define_comparison_mask(**kwargs)
        if mask is None:
            @tf.function
            def compare(x, y):
                try:
                    return x - y
                except Exception:
                    logging.info(f'x is {x}')
                    logging.info(f'y is {y}')
                    raise ValueError('Could not compare current and desired descriptor. ' +
                        f'This is maybe because of type mismatch, but most likely because ' + 
                        f'of shape mismatch. Either make sure the shapes match or ' + 
                        f'overwrite the descriptor subclass method define_comparison_mask ' + 
                        f'to define the behavior.')
            return compare
        @tf.function
        def compare_reduce_desired(smaller: tf.Tensor, larger: tf.Tensor) -> tf.Tensor:
            return tf.boolean_mask(larger, mask) - tf.reshape(smaller, [-1])
        if swap_args:
            @tf.function
            def compare_reduce_current(larger: tf.Tensor, smaller: tf.Tensor) -> tf.Tensor:
                return compare_reduce_desired(smaller, larger)
            return compare_reduce_current
        return compare_reduce_desired

    @classmethod
    def visualize(
            cls, 
            descriptor_value: np.ndarray, 
            save_as: str = None,
            descriptor_type: str = None):
        if isinstance(descriptor_value, tuple) or isinstance(descriptor_value, list):
            if save_as is not None:
                assert save_as.endswith('.png')
            for dim_number, dim_value in enumerate(descriptor_value):
                cls.visualize_slice(
                        dim_value,
                        save_as=f'{save_as[:-4]}_dimension_{dim_number+1}.png' if save_as is not None else None,
                        descriptor_type=descriptor_type)
        else:
            cls.visualize_slice(descriptor_value, save_as=save_as, descriptor_type=descriptor_type)


    @classmethod
    def visualize_slice(
            cls, 
            descriptor_value: np.ndarray, 
            save_as: str = None,
            descriptor_type: str = None):
        import matplotlib.pyplot as plt
        n_phases = descriptor_value.shape[0]
        mg_levels = descriptor_value.shape[1]
        fig, axs = plt.subplots(n_phases, mg_levels, sharex=True, sharey=True, squeeze=False)
        for n_phase in range(n_phases):
            for mg_level in range(mg_levels):
                cls.visualize_subplot(
                        descriptor_value[n_phase, mg_level],
                        axs[n_phase, mg_level],
                        descriptor_type=descriptor_type,
                        mg_level=mg_level,
                        n_phase=n_phase)
        plt.tight_layout()
        if save_as:
            logging.info(f'saving image as {save_as}')
            plt.savefig(save_as, dpi=600, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
    @classmethod
    def visualize_subplot(
            cls,
            descriptor_value: np.ndarray,
            ax,
            descriptor_type: str = None,
            mg_level: int = None,
            n_phase: int = None):
        import matplotlib.pyplot as plt
        x = descriptor_value.flatten()
        if descriptor_value.size < 10:
            ax.bar(np.arange(len(x)), x)
        else:
            area = descriptor_value.size
            for height in reversed(range(1, int(np.sqrt(area)) + 1)):
                width = area // height
                if width * height == area:
                    break
            x = x.reshape((height, width))
            ax.imshow(x, cmap='cividis')
        ax.set_title(f'{descriptor_type}: l={mg_level}, p={n_phase}')

