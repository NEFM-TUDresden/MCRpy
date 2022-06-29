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

import argparse
import pickle
import os
import logging
from typing import Dict, Tuple, Union

import numpy as np
import tensorflow as tf
try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except:
    pass

from mcrpy.src import log
from mcrpy.src import fileutils
from mcrpy.src import descriptor_factory
from mcrpy.src import DMCR
from mcrpy.src.Settings import ReconstructionSettings
from mcrpy.src.descriptor_factory import descriptor_choices


def main(args):
    """Main function for reconstruction script. This wraps some i/o around the reconstruct() function in order to make it usable
    via a command line interface and via a GUI. When mcrpy is used as a Python module directly, this main function is not called."""
    # prepare reconstruction
    args.target_folder = fileutils.create_target_folder(args)
    descriptor_dict = fileutils.read_descriptor(args.descriptor_filename)
    desired_shape = (args.extent_x, args.extent_y, args.extent_z) if args.extent_z not in {None, 0, 1} else (args.extent_x, args.extent_y)
    del args.extent_x
    del args.extent_y
    del args.extent_z
    initial_microstructure = None if args.initial_microstructure_file is None \
            else fileutils.read_ms(args.initial_microstructure_file)
    del args.initial_microstructure_file
    settings = ReconstructionSettings(**vars(args))

    # reconstruct
    convergence_data, last_frame = reconstruct(
            descriptor_dict, 
            desired_shape, 
            settings=settings,
            initial_microstructure=initial_microstructure)

    # prepare saving results
    information_additives = '' if settings.information is None else '_' + settings.information
    foldername = settings.target_folder if settings.target_folder is not None else ''

    # add settings to convergence data
    convergence_data['settings'] = settings

    # save convergence data
    save_convergence_data(convergence_data, foldername, information_additives=information_additives)

    # save last state separately
    save_last_state(last_frame, foldername, information_additives=information_additives)

def save_convergence_data(convergence_data: Dict[str, Any], foldername: str, information_additives: str = ''):
    filename = 'convergence_data{}.pickle'.format(information_additives)
    convergence_data_filename = os.path.join(foldername, filename)
    with open(convergence_data_filename, 'wb') as f:
        pickle.dump(convergence_data, f, protocol=4)

def save_last_state(last_state: np.ndarray, foldername: str, information_additives: str = ''):
    filename = 'last_frame{}.npy'.format(information_additives)
    np_data_filename = os.path.join(foldername, filename)
    save_microstructure(np_data_filename, last_state)

def save_microstructure(filename: str, ms: np.ndarray):
    assert filename.endswith('.npy')
    np.save(filename, ms)
    

def reconstruct(
        descriptor_dict: Dict[str, Union[np.ndarray, Tuple[np.ndarray]]],
        desired_shape: Tuple[int],
        settings: ReconstructionSettings = None,
        initial_microstructure: np.ndarray = None):
    if settings is None:
        settings = ReconstructionSettings()
    if any(s_i in {0, 1, None} for s_i in desired_shape):
        raise ValueError(f'Provided desired_shape {desired_shape} should not contain ' +
            'entries that are 0, 1 or None.')
    if len(desired_shape) not in {2, 3}:
        raise ValueError(f'Provided desired shape {desired_shape} has ' +
            f'{len(desired_shape)} entries for {len(desired_shape)} microstructure ' +
            f'dimensions, but only 2- and 3-dimensional microstructures are supported.')
    if initial_microstructure is not None:
        if 1 in initial_microstructure.shape:
            raise ValueError('Provided initial_microstructure np.ndarray has ' + 
                f'shape {initial_microstructure.shape}, which contains ones. ' + 
                f'The initial_microstructure shape should not contain ones.')
        if len(initial_microstructure.shape) not in {2, 3}:
            raise ValueError('Provided initial_microstructure np.ndarray has ' +
                f'shape {initial_microstructure.shape} and is therefore ' +
                f'{len(initial_microstructure.shape)}-dimensional, but it ' +
                f'should be 2- or 3-dimensional.')
        settings.initial_microstructure = np.array([fileutils.encode_ms(
                initial_microstructure, use_multiphase=settings.use_multiphase)]) 
    if settings.use_multigrid_reconstruction and not settings.use_multigrid_descriptor:
        raise ValueError('Provided settings.use_multigrid_reconstruction is True, ' +
                f'but settings.use_multigrid_descriptor is False. Cannot do ' +
                f'multigrid reconstruction withour mulrigrid descriptors, ' +
                f'although multigrid descriptors withour multigrid reconstruction is ' +
                f'possible and can be reasonable. Please fix the settings. Also, ' +
                f'if you want multigrid reconstruction, make sure the descriptor you ' +
                f'start with is a multigrid descriptor.')
    if settings.grey_values and settings.use_multiphase:
        raise ValueError('Cannot set both grey_values and use_multiphase!')

    # set up files, folders and logging
    log.setup_logging(settings.target_folder, settings)
    settings.save_to = settings.target_folder 

    # flatten and weight the descriptor
    desired_descriptors = [descriptor_dict[d] for d in settings.descriptor_types]
    if settings.descriptor_weights is None:
        settings.descriptor_weights = [1.0] * len(settings.descriptor_types)

    # augment settings
    if 'VolumeFractions' in descriptor_dict:
        settings.volume_fractions = descriptor_dict['VolumeFractions']


    # run reconstruction
    diff_mcr = DMCR.DMCR(**vars(settings))
    convergence_data, last_frame = diff_mcr.reconstruct(desired_descriptors, desired_shape)
    del diff_mcr
    return convergence_data, last_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct a microstructure given a descriptor')
    parser.add_argument('--descriptor_filename', type=str, help='File of descriptor that to reconstruct from', default='')
    parser.add_argument('--initial_microstructure_file', type=str, help='Microstructure to initializa with. If None, random initialization is used', default=None)
    parser.add_argument('--data_folder', type=str, help='Results folder. If None, default with timestamp.', default='results')
    parser.add_argument('--descriptor_types', nargs='+', type=str, help='Descriptor types (list)', default=['Correlations'], choices=descriptor_choices)
    parser.add_argument('--descriptor_weights', nargs='+', type=float, help='Descriptor weights (list)', default=None)
    parser.add_argument('--extent_x', type=int, help='Extent in x-direction in pixels.', default=64)
    parser.add_argument('--extent_y', type=int, help='Extent in y-direction in pixels.', default=64)
    parser.add_argument('--extent_z', type=int, help='Extent in z-direction in pixels.', default=1)
    parser.add_argument('--optimizer_type', type=str, help='Optimizer type defined in optimizer plugins.', default='LBFGSB')
    parser.add_argument('--loss_type', type=str, help='Loss type defined in loss plugins.', default='MSE')
    parser.add_argument('--nl_method', type=str, help='Nonlinearity method.', default='relu')
    parser.add_argument('--gram_weights_filename', type=str, help='Gram weigths filename wo path.', default='vgg19_normalized.pkl')
    parser.add_argument('--learning_rate', type=float, help='Learning rate of optimizer.', default=0.01)
    parser.add_argument('--beta_1', type=float, help='beta_1 parameter of optimizer.', default=0.9)
    parser.add_argument('--beta_2', type=float, help='beta_2 parameter of optimizer.', default=0.999)
    parser.add_argument('--rho', type=float, help='rho parameter of optimizer.', default=0.9)
    parser.add_argument('--momentum', type=float, help='momentum parameter of optimizer.', default=0.0)
    parser.add_argument('--limit_to', type=int, help='Limit in pixels to which to limit the characterisation metrics.', default=16)
    parser.add_argument('--threshold_steepness', type=float, help='Steepness of soft threshold function. Regularisation parameter.', default=10)
    parser.add_argument('--initial_temperature', type=float, help='Initial temperature for annealing', default=0.004)
    parser.add_argument('--final_temperature', type=float, help='Final temperature for annealing', default=None)
    parser.add_argument('--cooldown_factor', type=float, help='Cooldown factor for annealing', default=0.9)
    parser.add_argument('--max_iter', type=int, help='Maximum number of iterations.', default=500)
    parser.add_argument('--convergence_data_steps', type=int, help='Each x steps write data', default=10)
    parser.add_argument('--outfile_data_steps', type=int, help='Each x steps write data to disk', default=None)
    parser.add_argument('--tolerance', type=float, help='Resonctruction tolerance.', default=1e-10)
    parser.add_argument('--oor_multiplier', type=float, help='Penalty weight for OOR in loss', default=1000.0)
    parser.add_argument('--phase_sum_multiplier', type=float, help='Penalty weight for phase sum in loss', default=1000.0)
    parser.add_argument('--precision', type=int, help='Precision. 64 recommended', default=64)
    parser.add_argument('--information', type=str, help='Information that is added to files that are written.', default=None)
    parser.add_argument('--stride', type=int, help='Stride (3D DiffMCR).', default=4)
    parser.add_argument('--logfile_name', type=str, help='Name of logfile w/o extension.', default='logfile')
    parser.add_argument('--logging_level', type=int, help='Logging level.', default=logging.INFO)
    parser.add_argument('--logfile_date', dest='logfile_date', action='store_true')
    parser.set_defaults(logfile_date=False)
    parser.add_argument('--grey_values', dest='grey_values', action='store_true')
    parser.set_defaults(grey_values=False)
    parser.add_argument('--no_multiphase', dest='use_multiphase', action='store_false')
    parser.set_defaults(use_multiphase=True)
    parser.add_argument('--no_multigrid_descriptor', dest='use_multigrid_descriptor', action='store_false')
    parser.set_defaults(use_multigrid_descriptor=True)
    parser.add_argument('--use_multigrid_reconstruction', dest='use_multigrid_reconstruction', action='store_true')
    parser.set_defaults(use_multigrid_reconstruction=False)
    args = parser.parse_args()
    main(args)
