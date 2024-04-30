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
from typing import Tuple

import numpy as np
import tensorflow as tf

from mcrpy.src.IndicatorFunction import IndicatorFunction

class Descriptor(ABC):
    is_differentiable = True
    default_weight = 1.0

    @classmethod
    def make_singlegrid_descriptor(
            cls,
            desired_shape_2d=None,
            desired_shape_extended=None,
            limit_to=8,
            **kwargs):
        singlephase_kwargs = {
                'desired_shape_2d': desired_shape_2d,
                'desired_shape_extended': desired_shape_extended,
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
            desired_shape_layer = tuple(s//pool_size for s in desired_shape_2d)
            singlephase_kwargs = {
                    'desired_shape_2d': desired_shape_layer,
                    'desired_shape_extended': (1, *desired_shape_layer, n_phases),
                    'limit_to': limit_to,
                    **kwargs
                    }
            singlephase_descriptors.append(cls.wrap_singlephase_descriptor(**singlephase_kwargs))
        else:
            logging.info('Could create all {} MG levels'.format(mg_levels))

        @tf.function
        def multigrid_descriptor(mg_input):
            mg_layers = []
            # print('######################## enter function #############################')
            # print(type(mg_input))
            for mg_level, singlephase_descriptor in enumerate(singlephase_descriptors):
                pool_size = 2**mg_level
                if isinstance(mg_input, IndicatorFunction):
                    mg_pool = IndicatorFunction(tf.nn.avg_pool2d(mg_input.x, [pool_size, pool_size], [pool_size, pool_size], 'VALID'))
                elif isinstance(mg_input, tf.Tensor):
                    mg_pool = tf.nn.avg_pool2d(mg_input, [pool_size, pool_size], [pool_size, pool_size], 'VALID')
                else:
                    raise ValueError('mg_input should be IndicatorFunction')
                mg_desc = singlephase_descriptor(mg_pool)
                mg_exp = tf.expand_dims(mg_desc, axis=0)
                mg_layers.append(mg_exp)
            outputs = tf.concat(mg_layers, axis=0) if len(mg_layers) > 1 else mg_layers[0]
            return outputs

        return multigrid_descriptor

    @classmethod
    def wrap_singlephase_descriptor(
            cls, 
            **kwargs):
        if cls.is_differentiable:
            return cls.make_singlephase_descriptor(**kwargs)
        compute_descriptor_np = cls.make_singlephase_descriptor(**kwargs)

        def compute_descriptor_tf(x: tf.Tensor) -> tf.Tensor:
            x_np = x.numpy()
            y_np = compute_descriptor_np(x_np)
            y_tf = tf.constant(y_np.astype(np.float64), dtype=tf.float64)
            return y_tf

        @tf.function 
        def compute_descriptor_compiled(x: tf.Tensor) -> tf.Tensor:
            py_descriptor = tf.py_function(func=compute_descriptor_tf, inp=[x], Tout=tf.float64)
            return py_descriptor

        return compute_descriptor_compiled

    @classmethod
    def make_singlephase_descriptor(
            cls, 
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
                    raise ValueError("""Could not compare current and desired descriptor. 
                        This is maybe because of type mismatch, but most likely because 
                        of shape mismatch. Either make sure the shapes match or 
                        overwrite the descriptor subclass method define_comparison_mask 
                        to define the behavior.""")
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
        if isinstance(descriptor_value, (tuple, list)):
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
        import matplotlib
        import matplotlib.pyplot as plt
        
        # matplotlib.rcParams.update({
        #     "pgf.texsystem":"pdflatex",
        #     'font.family': 'serif',
        #     'font.size': 10,
        #     'figure.titlesize': 'medium',
        #     'text.usetex':'True',
        #     'pgf.rcfonts':'False',
        #     "pgf.preamble": r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{mathrsfs}"
        #     })
        
        n_phases = descriptor_value.shape[0]
        mg_levels = descriptor_value.shape[1]
        fig, axs = plt.subplots(n_phases, mg_levels, squeeze=False)
        # fig, axs = plt.subplots(n_phases, mg_levels, sharex=True, sharey=True, squeeze=False)
        for n_phase in range(n_phases):
            for mg_level in range(mg_levels):
                cls.visualize_subplot(
                        descriptor_value[n_phase, mg_level],
                        axs[n_phase, mg_level],
                        descriptor_type=descriptor_type,
                        mg_level=mg_level,
                        n_phase=n_phase if n_phases > 1 else 1)
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

def make_image_padder(pad_x: int, pad_y: int):

    @tf.function
    def tile_img(img: tf.Tensor) -> tf.Tensor:
        """Tile an image. Needed for periodic boundary conditions in convolution."""
        img_tiled_x = tf.concat([img, img[:, :pad_x, :, :]], axis=1)
        img_tiled_xy = tf.concat([img_tiled_x, img_tiled_x[:, :, :pad_y, :]], axis=2)
        return img_tiled_xy
    return tile_img

def test_padding():
    import matplotlib.pyplot as plt

    ms = np.load('../../microstructures/pymks_ms_64x64.npy')
    ms = ms.reshape([1, *ms.shape, 1])
    padder = make_image_padder(15, 1)

    savefig = False

    plt.figure(figsize=(4, 4))
    plt.imshow(padder(ms)[0, :, :, 0])
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig('plot.png', dpi=600, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    

    plt.figure(figsize=(4, 4))
    plt.imshow(ms[0, :, :, 0])
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig('plot.png', dpi=600, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    

if __name__ == "__main__":
    test_padding()
