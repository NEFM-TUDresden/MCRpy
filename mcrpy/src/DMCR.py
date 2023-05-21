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

import contextlib
import logging
import copy
import os
import sys
from typing import List

import numpy as np
import scipy.ndimage as ndimg
import tensorflow as tf
with contextlib.suppress(Exception):
    tf.config.experimental.enable_tensor_float_32_execution(False)
from mcrpy.src import log
from mcrpy.src import loader
from mcrpy.src import descriptor_factory
from mcrpy.src import loss_factory
from mcrpy.src import optimizer_factory
from mcrpy.src import loss_computation
from mcrpy.src import profile
from mcrpy.src.Microstructure import Microstructure
from mcrpy.src.MutableMicrostructure import MutableMicrostructure
from mcrpy.src.Symmetry import Symmetry
from mcrpy.descriptors.OrientationDescriptor import OrientationDescriptor
from mcrpy.descriptors.MultiPhaseDescriptor import MultiPhaseDescriptor

class DMCR:
    def __init__(self,
            descriptor_types: List[str] = None,
            descriptor_weights: List[float] = None,
            loss_type: str = 'MSE',
            optimizer_type: str = 'LBFGSB',
            convergence_data_steps: int = 10,
            outfile_data_steps: int = None,
            save_to: str = 'results',
            use_multigrid_reconstruction: bool = False,
            use_multigrid_descriptor: bool = False,
            use_multiphase: bool = False,
            limit_to: int = 8,
            n_phases: int = None,
            minimal_resolution: int = None,
            information: str = None,
            tvd: float = None,
            max_iter: int = 500,
            phase_sum_multiplier: float = 1000.0,
            oor_multiplier: float = 1000.0,
            volume_fractions: np.ndarray = None,
            greedy: bool = False,
            batch_size: int = 1,
            symmetry: Symmetry = None,
            initial_microstructure: Microstructure = None,
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
        if descriptor_types is None:
            descriptor_types = ['Correlations']
        self.descriptor_types = descriptor_types
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.max_iter = max_iter
        self.convergence_data_steps = convergence_data_steps
        self.outfile_data_steps = outfile_data_steps if outfile_data_steps is not None else np.inf
        self.save_to = save_to
        self.use_multigrid_reconstruction = use_multigrid_reconstruction
        self.use_multigrid_descriptor = use_multigrid_descriptor
        self.use_multiphase = use_multiphase
        self.information_additives = '' if information is None else f'_{information}'
        self.tvd = tvd
        self.volume_fractions = volume_fractions
        self.oor_multiplier = oor_multiplier
        self.phase_sum_multiplier = phase_sum_multiplier
        self.limit_to = limit_to
        self.n_phases = n_phases
        self.greedy = greedy
        self.symmetry = symmetry
        self.batch_size = batch_size
        self.initial_microstructure = initial_microstructure
        self.minimal_resolution = minimal_resolution if minimal_resolution is not None else limit_to
        tf.keras.backend.set_floatx("float64")
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

        # check if use_orientations
        orientation_descriptors = [issubclass(descriptor_factory.descriptor_classes[descriptor_type], OrientationDescriptor) 
                for descriptor_type in descriptor_types]
        self.use_orientations = any(orientation_descriptors)
        if self.use_orientations:
            assert all(orientation_descriptors), "If one descriptor is an OrientationDescriptor, then all should be"
            assert not use_multiphase, "Orientations and multiphase cannot be used together"

        # use default descriptor weights if no descriptor weights are given
        if descriptor_weights is None:
            descriptor_weights = [
                    descriptor_factory.descriptor_classes[descriptor_type].default_weight
                    for descriptor_type in descriptor_types]
        if len(descriptor_types) != len(descriptor_weights):
            raise ValueError('descriptor_types and descriptor_weights do not have the same length!')
        self.descriptor_weights = descriptor_weights

        # find which descriptors are MultiPhaseDescriptors
        self.descriptor_is_multiphase = [issubclass( 
            descriptor_factory.descriptor_classes[descriptor_type], MultiPhaseDescriptor) 
            for descriptor_type in self.descriptor_types]

        # initialize some values for later assertion
        self.last_lme_iter = 0
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
                'n_phases': self.n_phases,
                'symmetry': self.symmetry,
                **self.kwargs
                }
        if self.non_cubic_3d:
            self.descriptors = []
            desired_shapes_2d = [
                    (self.desired_shape_ms[1], self.desired_shape_ms[2]),
                    (self.desired_shape_ms[0], self.desired_shape_ms[2]),
                    (self.desired_shape_ms[0], self.desired_shape_ms[1]),
                    ]
            for desired_shape_2d in desired_shapes_2d:
                descriptor_kwargs['desired_shape_2d'] = desired_shape_2d
                self.descriptors.append([descriptor_factory.create(
                    descriptor_type, arguments=descriptor_kwargs, 
                    assert_differentiable=self.is_gradient_based) 
                    for descriptor_type in self.descriptor_types])
        else:
            self.descriptors = [descriptor_factory.create(
                descriptor_type, arguments=descriptor_kwargs, 
                assert_differentiable=self.is_gradient_based) 
                for descriptor_type in self.descriptor_types]

        desired_descriptor_shapes = [
            desired_value[0].shape[2:]
            if self.anisotropic
            else desired_value.shape[2:]
            for desired_value in self.desired_descriptor
        ]
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

    def validate_n_phases(self):
        for desired_value, d_is_multiphase in zip(
                self.desired_descriptor, self.descriptor_is_multiphase):
            if d_is_multiphase:
                continue
            if self.anisotropic:
                desired_descriptor_phases = desired_value[0].shape[0]
            else:
                desired_descriptor_phases = desired_value.shape[0]
            assert desired_descriptor_phases == self.n_phases
        if self.n_phases > 2 and not self.use_multiphase:
            raise ValueError(f'n_phases = {self.n_phases} > 2 but use_multiphase is False.')

    @log.log_this
    def assign_desired_shape(self, input_shape):
        """Assign the shape of the microstructure to reconstruct. Also initializes the microstructure and sets up the descriptor."""
        assert len(input_shape) in {2, 3}
        self.desired_shape_ms = input_shape
        self.validate_n_phases()
        if self.use_orientations:
            self.desired_shape_extended = (1, *input_shape, 3)
        else:
            self.desired_shape_extended = (
                    1, *input_shape, self.n_phases if self.use_multiphase else 1)
        self.non_cubic_3d = len(set(self.desired_shape_ms)) > 1 and len(self.desired_shape_ms) == 3

    def initialize_microstructure(self, previous_solution: Microstructure = None):
        """Initialize the ms by sampling randomly or upsampling previous solution."""
        loc = 0.5
        ms_class = MutableMicrostructure if  optimizer_factory.optimizer_classes[self.optimizer_type].swaps_pixels else Microstructure
        if previous_solution is None:
            img = np.random.normal(loc=loc, scale=0.1, size=self.desired_shape_extended)
        else:
            img = self.resample_microstructure(previous_solution)
        np.clip(img, 0, 1, out=img)
        self.ms = ms_class(img.reshape(self.desired_shape_extended), use_multiphase=self.use_multiphase, skip_encoding=True, trainable=self.is_gradient_based, symmetry=self.symmetry)

    def resample_microstructure(self, ms: Microstructure, zoom: float = None):
        """Upsample a MS."""
        if zoom is not None:
            zoom_factor = (1, zoom, zoom, zoom, 1) if ms.is_3D else (1, zoom, zoom, 1)
        else:
            zoom_factor = tuple(
                des / cur
                for des, cur in zip(self.desired_shape_extended, ms.x_shape)
            )
        return ndimg.zoom(ms.x.numpy(), zoom_factor, order=1)

    def reconstruction_callback(self, n_iter: int, l: float, ms: Microstructure, force_save: int = False, safe_mode: bool = False):
        """Function to call every iteration for monitoring convergence and storing results. Technically not a callback function."""
        self.convergence_data['line_data'].append((n_iter, l))
        if n_iter % self.convergence_data_steps == 0 or force_save:
            tf.print('Iteration', n_iter, 'of', self.max_iter, ':', l, output_stream=sys.stdout)
            self.convergence_data['scatter_data'].append((n_iter, l))
            # self.convergence_data['raw_data'].append([self.resample_microstructure(ms, zoom=self.pool_size)])
            self.convergence_data['raw_data'].append(copy.deepcopy(self.ms))
        if n_iter % self.outfile_data_steps == 0 and (
                n_iter > 0 or self.outfile_data_steps < np.inf):
            foldername = self.save_to if self.save_to is not None else ''
            n_digits = len(str(self.max_iter))
            filename = f'ms{self.information_additives}_level_{self.mg_level}_iteration_{str(n_iter).zfill(n_digits)}.npy'
            outfile = os.path.join(foldername, filename)
            ms.to_npy(outfile)


    @log.log_this
    @profile.maybe_profile('logdir')
    def reconstruct(self, desired_descriptor, desired_shape):
        """Start reconstruction. The desired descriptor and desired shape should be assigned before. See constructor docstring for more details."""
        # assign desired descriptor
        self.assign_desired_descriptor(desired_descriptor)

        mg_levels = self._determine_mg_levels(desired_descriptor, desired_shape)

        previous_solution = self.initial_microstructure
        for mg_level in reversed(range(mg_levels)):
            self.setup_optimization(desired_shape, mg_levels, previous_solution, mg_level)
            last_iter = None
            try:
                last_iter = self._optimize(last_iter)
            except KeyboardInterrupt:
                logging.info('KeyboardInterrupt, stop opt and continue')
                last_iter = self.max_iter
                break
            previous_solution = self.ms
        assert last_iter is not None
        
        # add last state
        self.reconstruction_callback(
                last_iter + 1, self.opt.current_loss, self.ms, force_save=True, safe_mode=True)

        # convert convergence data to np
        for k, v in self.convergence_data.items():
            self.convergence_data[k] = np.array(v)

        # return result
        return self.convergence_data, self.ms

    def setup_optimization(self, desired_shape, mg_levels, previous_solution, mg_level):
        self.mg_level = mg_level
        self.pool_size = 2**mg_level

        # assign desired shape
        self.assign_desired_shape(
                tuple(s_i // self.pool_size for s_i in desired_shape)
            )

        # init ms
        self.initialize_microstructure(previous_solution)

        # set up descriptor
        self.setup_descriptor()

        # assert that we are ready to reconstruct
        self._assert_initialization()

        # create loss
        self._create_loss(mg_levels, mg_level) 

        # create optimizer
        self._create_optimizer()

    def _optimize(self, last_iter):
        print('start optimization')
        last_iter = self.opt.optimize(self.ms, restart_from_niter=last_iter)
        return last_iter

    def _assert_initialization(self):
        assert self.desired_descriptor is not None
        assert self.desired_shape_ms is not None
        assert self.desired_shape_extended is not None
        assert self.anisotropic is not None

    def _determine_mg_levels(self, desired_descriptor, desired_shape):
        if self.use_multigrid_reconstruction:
            limitation_factor = min(s_i / self.minimal_resolution for s_i in desired_shape)
            logging.info(f'self.minimal_resolution = {self.minimal_resolution}')
            logging.info(f'desired_shape = {desired_shape}')
            logging.info(f'limitation_factor = {limitation_factor}')
            mg_levels_ms = int(np.floor(np.log(limitation_factor) / np.log(2)))
            mg_levels_d = min(d_des[0].shape[1] for d_des in desired_descriptor) if self.anisotropic else min(d_des.shape[1] for d_des in desired_descriptor)
            mg_levels = min(mg_levels_d, mg_levels_ms)
            logging.info(f'preparing {mg_levels} sequential MG levels')
        else:
            mg_levels = 1
        return mg_levels

    def _create_optimizer(self):
        opt_kwargs = {
                    'max_iter': self.max_iter,
                    'desired_shape_extended': self.desired_shape_extended,
                    'callback': self.reconstruction_callback,
                    'loss': self.loss,
                    'use_multiphase': self.use_multiphase,
                    'is_3D': self.ms.is_3D,
                    'use_orientations': self.use_orientations,
                    **self.kwargs,
                    }
        self.opt = optimizer_factory.create(self.optimizer_type, arguments=opt_kwargs)

        self.opt.set_call_loss(loss_computation.make_call_loss(
                self.loss, self.ms, self.is_gradient_based, sparse=self.opt.is_sparse,
                greedy=self.greedy, batch_size = self.batch_size))

        # pass vf information extra if needed
        if optimizer_factory.optimizer_classes[self.optimizer_type].is_vf_based:
            if self.volume_fractions is None:
                raise ValueError(f'The chosen optimizer {self.optimizer_type} ' +
                    'requires the volume fraction to be given, but no volume fractions ' +
                    'were found in the descriptor.')
            self.opt.set_volume_fractions(self.volume_fractions)

    def _create_loss(self, mg_levels, mg_level):
        loss_kwargs = {
                    'descriptor_list': self.descriptors,
                    'desired_descriptor_list': self.desired_descriptor,
                    'descriptor_weights': self.descriptor_weights,
                    'descriptor_comparisons': self.descriptor_comparisons,
                    'tvd': self.tvd,
                    'anisotropic': self.anisotropic,
                    'phase_sum_multiplier': self.phase_sum_multiplier,
                    'oor_multiplier': self.oor_multiplier,
                    'mg_level': mg_level,
                    'mg_levels': mg_levels,
                    'use_multiphase': self.use_multiphase,
                    'desired_shape_extended': self.desired_shape_extended,
                    'descriptor_is_multiphase': self.descriptor_is_multiphase,
                    'use_orientations': self.use_orientations,
                    **self.kwargs,
                    }
        self.loss = loss_factory.create(self.loss_type, non_cubic_3d=self.non_cubic_3d, arguments=loss_kwargs)
