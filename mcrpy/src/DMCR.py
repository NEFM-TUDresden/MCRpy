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

import itertools
import logging
import os
import sys
import pickle
from typing import List

import numpy as np
import scipy.ndimage as ndimg
import tensorflow as tf
try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except:
    pass

from mcrpy.src import log
from mcrpy.src import loader
from mcrpy.src import descriptor_factory
from mcrpy.src import loss_factory
from mcrpy.src import optimizer_factory
from mcrpy.src import numerical_precision
from mcrpy.src import loss_computation
from mcrpy.src import fileutils

class DMCR:
    def __init__(self,
            descriptor_types: List[str] = ['Correlations'],
            descriptor_weights: List[float] = [1.0],
            loss_type: str = 'MSE',
            optimizer_type: str = 'LBFGSB',
            precision: int = 64,
            convergence_data_steps: int = 10,
            outfile_data_steps: int = None,
            save_to: str = 'results',
            use_multigrid_reconstruction: bool = False,
            use_multigrid_descriptor: bool = False,
            use_multiphase: bool = False,
            limit_to: int = 8,
            minimal_resolution: int = None,
            information: str = None,
            tvd: float = None,
            stride: int = 64,
            max_iter: int = 500,
            phase_sum_multiplier: float = 1000.0,
            oor_multiplier: float = 1000.0,
            volume_fractions: np.ndarray = None,
            initial_microstructure: np.ndarray = None,
            **kwargs):
        """Initializer for differentiable microstructure characterisation and reconstruction (DMCR).
        DMCR formulates microstructure reconstruction as a differentiable optimization problem.
        The difference between the current and desired descriptor is minimized using gradient-based optimization.
        This means that the characterization functions need to be formulated in a differentiable manner.
        The gradients are then computed vie TensorFlow backpropagation.
        This automatically handles just-in-time compilation as well.
        The ABC for a descriptor is defined in descriptors/Descriptor.py .
        Any class that implements this protocol is a valid descriptor, regardless of inheritance.
        Descriptors are located in the descriptors folder and are imported dynamically.
        This means that for addingn a new descriptor to the package, no code needs to be changed in the package.
        Not even import statements need to be added.
        The only thing to do is to add the descriptor file at the right place and append it to the descriptor_types list.
        This can be done e.g. as an additional command line argument to matching.py .
        The same modular design is made for the loss_type and the optimizer_type.
        For this extensibility, the DMCR constructor accepts generic **kwargs.
        These are passed to the descriptor, loss and optimizer constructors and allow flexible extension.
        The downside is that sensible defaults need to be set.
        If this is not possible, please implement assertions.
        Currently, all descriptors assume periodic boundary conditions.
        This is bad for prescribing the desired value but good for numerical simulation with periodic boundaries.
        For more information, read the initial 2D paper:
            Seibert, Ambati, Raßloff, Kästner, Reconstructing random heterogeneous media through differentiable
            optimization, 2021
        and the 3D extension:
            Seibert, Ambati, Raßloff, Kästner, Descriptor-based reconstruction of three-dimensional microstructures
            through gradient-based optimization, 2021
        """
        if not len(descriptor_types) == len(descriptor_weights):
            raise ValueError('descriptor_types and descriptor_weights do not have the same length!')

        # assign values
        self.descriptor_types = descriptor_types
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.descriptor_weights = descriptor_weights
        self.max_iter = max_iter
        self.convergence_data_steps = convergence_data_steps
        self.outfile_data_steps = outfile_data_steps if outfile_data_steps is not None else np.inf
        self.save_to = save_to
        self.use_multigrid_reconstruction = use_multigrid_reconstruction
        self.use_multigrid_descriptor = use_multigrid_descriptor
        self.use_multiphase = use_multiphase
        self.information_additives = '' if information is None else '_' + information
        self.tvd = tvd
        self.stride = stride
        self.volume_fractions = volume_fractions
        self.oor_multiplier = oor_multiplier
        self.phase_sum_multiplier = phase_sum_multiplier
        self.limit_to = limit_to
        self.initial_microstructure = initial_microstructure
        self.minimal_resolution = minimal_resolution if minimal_resolution is not None else limit_to
        numerical_precision.set_precision(self, precision)
        self.kwargs = kwargs
        self.convergence_data = {
                'raw_data': [],
                'line_data': [],
                'scatter_data': []
                }

        if not self.use_multiphase:
            self.phase_sum_multiplier = 0.0

        # load modules
        loader.load_plugins([f'mcrpy.descriptors.{descriptor_type}' 
            for descriptor_type in self.descriptor_types])
        loader.load_plugins([f'mcrpy.losses.{loss_type}'])
        loader.load_plugins([f'mcrpy.optimizers.{optimizer_type}'])

        # check if optimizer gradient-based
        self.is_gradient_based = optimizer_factory.optimizer_classes[optimizer_type].is_gradient_based

        # initialize some values for later assertion
        self.desired_descriptor = None
        self.desired_shape_ms = None
        self.desired_shape_extended = None
        self.loss = None
        self.anisotropic = None

    @log.log_this
    def setup_descriptor(self):
        """Set up self.descriptor by preparing the args and calling the factory."""
        descriptor_kwargs = {
                'limit_to': self.limit_to,
                'use_multigrid_descriptor': self.use_multigrid_descriptor,
                'use_multiphase': self.use_multiphase,
                'desired_shape_2d': (self.desired_shape_ms[0], self.desired_shape_ms[1]),
                'desired_shape_extended': self.desired_shape_extended,
                'tf_dtype': self.tf_dtype,
                'np_dtype': self.np_dtype,
                **self.kwargs
                }
        self.descriptors = [descriptor_factory.create(
            descriptor_type, arguments=descriptor_kwargs, 
            assert_differentiable=self.is_gradient_based) 
            for descriptor_type in self.descriptor_types]
        desired_descriptor_shapes = [desired_value.shape[2:] 
                if not self.anisotropic else desired_value[0].shape[2:]
                for desired_value in self.desired_descriptor]
        self.descriptor_comparisons = [descriptor_factory.get_class(
            descriptor_type).make_comparison(
                desired_descriptor_shape=desired_descriptor_shape, 
                limit_to=self.limit_to,
                **self.kwargs) for descriptor_type, desired_descriptor_shape in zip(
                    self.descriptor_types, desired_descriptor_shapes)]


    @log.log_this
    def assign_desired_descriptor(self, desired_descriptor):
        """Assign desired_descriptor."""
        type_list = [type(d) for d in desired_descriptor]
        self.anisotropic = tuple in type_list or list in type_list
        self.desired_descriptor = desired_descriptor


    @log.log_this
    def assign_desired_shape(self, input_shape):
        """Assign the shape of the microstructure to reconstruct. Also initializes the microstructure and sets up the descriptor."""
        assert len(input_shape) in {2, 3}
        self.ms_is_3d = len(input_shape) == 3
        self.desired_shape_ms = input_shape
        desired_descriptor_phases = [desired_value.shape[0]
                if not self.anisotropic else desired_value[0].shape[0]
                for desired_value in self.desired_descriptor]
        if np.unique(np.array(desired_descriptor_phases)).size != 1:
            raise ValueError('Not all descriptors have been computed for the same ' + 
                    f'number of phases!')
        self.n_phases = desired_descriptor_phases[0]
        if self.n_phases > 2 and not self.use_multiphase:
            raise ValueError(f'n_phases = {self.n_phases} > 2 but use_multiphase is False.')
        self.desired_shape_extended = (
                1, *input_shape, self.n_phases if self.use_multiphase else 1)

    def initialize_microstructure(self, previous_solution: np.ndarray = None):
        """Initialize the ms by sampling randomly or upsampling previous solution."""
        if previous_solution is None:
            img = np.random.normal(loc=0.5, scale=0.1, size=self.desired_shape_extended)
        else:
            img = self.resample_microstructure(previous_solution)
        np.clip(img, 0, 1, out=img)
        self.x = tf.Variable(img.reshape(self.desired_shape_extended), name='ms', dtype=self.tf_dtype, trainable=True)

    def resample_microstructure(self, ms: np.ndarray, zoom: float = None):
        """Upsample a MS."""
        if zoom is not None:
            zoom_factor = (1, zoom, zoom, zoom, 1) if self.ms_is_3d else (1, zoom, zoom, 1)
        else:
            zoom_factor = tuple([des / cur for des, cur in zip(self.desired_shape_extended, ms.shape)])
        x_upsampled = ndimg.zoom(ms, zoom_factor, order=1)
        return x_upsampled

    def reconstruction_callback(self, n_iter, l, x, force_save=False):
        """Function to call every iteration for monitoring convergence and storing results. Technically not a callback function."""
        self.convergence_data['line_data'].append((n_iter, l))
        if n_iter % self.convergence_data_steps == 0 or force_save:
            tf.print('Iteration', n_iter, 'of', self.max_iter, ':', l, output_stream=sys.stdout)
            self.convergence_data['scatter_data'].append((n_iter, l))
            self.convergence_data['raw_data'].append([self.resample_microstructure(x.numpy(), zoom=self.pool_size)])
        if n_iter % self.outfile_data_steps == 0 and (
                n_iter > 0 or self.outfile_data_steps < np.inf):
            foldername = self.save_to if self.save_to is not None else ''
            n_digits = len(str(self.max_iter))
            filename = f'ms{self.information_additives}_level_{self.mg_level}_iteration_{str(n_iter).zfill(n_digits)}.npy'
            outfile = os.path.join(foldername, filename)
            np.save(outfile, fileutils.decode_ms(x.numpy()))


    @log.log_this
    def reconstruct(self, desired_descriptor, desired_shape):
        """Start reconstruction. The desired descriptor and desired shape should be assigned before. See constructor docstring for more details."""
        # assign desired descriptor
        self.assign_desired_descriptor(desired_descriptor)

        if self.use_multigrid_reconstruction:
            limitation_factor = min(s_i / self.minimal_resolution for s_i in desired_shape)
            logging.info(f'self.minimal_resolution = {self.minimal_resolution}')
            logging.info(f'desired_shape = {desired_shape}')
            logging.info(f'limitation_factor = {limitation_factor}')
            mg_levels = int(np.floor(np.log(limitation_factor) / np.log(2)))
            logging.info(f'preparing {mg_levels} sequential MG levels')
        else:
            mg_levels = 1

        previous_solution = self.initial_microstructure
        for mg_level in reversed(range(mg_levels)):
            self.mg_level = mg_level
            self.pool_size = 2**mg_level

            # assign desired shape
            self.assign_desired_shape(tuple([s_i // self.pool_size for s_i in desired_shape]))

            # init ms
            self.initialize_microstructure(previous_solution)

            # set up descriptor
            self.setup_descriptor()

            # assert that we are ready to reconstruct
            assert self.desired_descriptor is not None
            assert self.desired_shape_ms is not None
            assert self.desired_shape_extended is not None
            assert self.anisotropic is not None

            # create loss
            loss_kwargs = {
                    'descriptor_list': self.descriptors,
                    'desired_descriptor_list': self.desired_descriptor,
                    'descriptor_weights': self.descriptor_weights,
                    'descriptor_comparisons': self.descriptor_comparisons,
                    'tvd': self.tvd,
                    'tf_dtype': self.tf_dtype,
                    'anisotropic': self.anisotropic,
                    'phase_sum_multiplier': self.phase_sum_multiplier,
                    'oor_multiplier': self.oor_multiplier,
                    'mg_level': mg_level,
                    'mg_levels': mg_levels,
                    'use_multiphase': self.use_multiphase,
                    'desired_shape_extended': self.desired_shape_extended,
                    **self.kwargs,
                    }
            self.loss = loss_factory.create(self.loss_type, arguments=loss_kwargs) 

            # create optimizer
            opt_kwargs = {
                    'max_iter': self.max_iter,
                    'desired_shape_extended': self.desired_shape_extended,
                    'np_dtype': self.np_dtype,
                    'callback': self.reconstruction_callback,
                    'loss': self.loss,
                    'use_multiphase': self.use_multiphase,
                    **self.kwargs,
                    }
            self.opt = optimizer_factory.create(self.optimizer_type, arguments=opt_kwargs)

            self.opt.set_call_loss(loss_computation.make_call_loss(
                self.loss, self.desired_shape_ms, 
                self.n_phases if self.use_multiphase else 1, 
                self.np_dtype,
                self.is_gradient_based,
                stride=self.stride))

            # pass vf information extra if needed
            if optimizer_factory.optimizer_classes[self.optimizer_type].is_vf_based:
                if self.volume_fractions is None:
                    raise ValueError(f'The chosen optimizer {self.optimizer_type} ' +
                    'requires the volume fraction to be given, but no volume fractions ' +
                    'were found in the descriptor.')
                self.opt.set_volume_fractions(self.volume_fractions)

            # optimize
            try:
                last_iter = self.opt.optimize(self.x)
            except KeyboardInterrupt:
                logging.info(f'KeyboardInterrupt, stop opt and continue')
                last_iter = self.max_iter

            previous_solution = self.x.numpy()

        # add last state
        self.reconstruction_callback(
                last_iter + 1, self.opt.current_loss, self.x, force_save=True)
        last_state = fileutils.decode_ms(self.x.numpy()).reshape(self.desired_shape_ms)
        
        # convert convergence data to np
        for k, v in self.convergence_data.items():
            self.convergence_data[k] = np.array(v)

        # save result
        return self.convergence_data, last_state
