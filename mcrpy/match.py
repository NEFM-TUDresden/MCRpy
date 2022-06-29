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
import os
import logging
import subprocess
from typing import List, Tuple

import numpy as np
import tensorflow as tf
try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except:
    pass

from mcrpy.characterize import characterize, save_characterization
from mcrpy.reconstruct import reconstruct, save_convergence_data, save_last_state
from mcrpy.src import log
from mcrpy.src import fileutils
from mcrpy.src.Settings import MatchingSettings, CharacterizationSettings, ReconstructionSettings, select_subsettings
from mcrpy.src.descriptor_factory import descriptor_choices

def main(args: argparse.Namespace):
    """Main function for matching script. This wraps some i/o around the match() function in order to make it usable
    via a command line interface and via a GUI. When mcrpy is used as a Python module directly, this main function is not called."""
    initial_microstructure = None if args.initial_microstructure_file is None \
        else fileutils.read_ms(args.initial_microstructure_file)
    del args.initial_microstructure_file
    add_dimension = args.add_dimension
    del args.add_dimension
    settings = MatchingSettings(**vars(args))

    settings.target_folder = fileutils.create_target_folder(settings)
    microstructure_filename = fileutils.copy_ms_to_target(settings.target_folder, settings)
    microstructure = fileutils.read_ms(microstructure_filename)

    # run matching
    characterization, convergence_data, last_frame = match(
            microstructure,
            add_dimension=add_dimension,
            initial_microstructure=initial_microstructure,
            settings=settings)

    # add settings to convergence data
    convergence_data['settings'] = settings

    # prepare saving results
    info_file_additive = '.'.join(
            os.path.split(microstructure_filename)[-1].split('.')[:-1]) + '_' + (
                    settings.information + '_' if settings.information is not None else '')
    information_additives = '' if settings.information is None else '_' + settings.information
    foldername = settings.target_folder if settings.target_folder is not None else ''

    # save characterization
    save_characterization(characterization, info_file_additive, settings.target_folder)

    # save convergence data
    save_convergence_data(convergence_data, foldername, information_additives=information_additives)

    # save last state separately
    save_last_state(last_frame, foldername, information_additives=information_additives)

def match(
        microstructure: np.ndarray,
        add_dimension: int = None,
        initial_microstructure: np.ndarray = None,
        settings: MatchingSettings = None):
    """Match a microstructure by characterizing it and immediately reconstructing from the same descriptor. This is not really
    an application scenario, but helps in research and development. Also, it allows to check whether a certain set of descriptors
    and settings are suitable for reconstruction a certain type of microstructure. Under the hood, this function does nothing but
    calling characterize() and reconstruct() after each other. For an understanding of the settings for characterization and
    reconstruction, read the documentation of these two functions. As an extra argument, add_dimension can be passed. If this is
    set to an integer, the reconstruction adds add_dimension pixels as a third dimension to reconstruct a 3d microstructure from
    descriptors that are computed on a 2d slice. For more information on the 2D-to-3D reconstruction, read Seibert et al,
    Descriptor-based reconstruction of three-dimensional microstructures through gradient-based optimization, Acta Materialia,
    2022."""
    if settings is None:
        settings = MatchingSettings()
    if settings.grey_values and settings.use_multiphase:
        raise ValueError('Cannot set both grey_values and use_multiphase!')

    # prepare characterization
    characterization_settings = select_subsettings(settings, CharacterizationSettings)

    # run characterization
    characterization = characterize(microstructure, settings=characterization_settings)

    desired_shape = microstructure.shape
    if add_dimension is not None:
        if len(desired_shape) == 3:
            raise ValueError('Provided settings for add_dimension, but microstructure is ' +
                'already 3D.')
        elif len(desired_shape) == 2:
            desired_shape = tuple(list(desired_shape) + [add_dimension])
    reconstruction_settings = select_subsettings(settings, ReconstructionSettings)
    if not settings.grey_values:
        phases = np.unique(microstructure)
        if not all(np.array([i for i in range(len(phases))]) == phases):
            raise ValueError('Phase numbering should start with zero and increase with ' + 
                'increments of one.')

    # run reconstruction
    convergence_data, last_frame = reconstruct(
            characterization,
            desired_shape,
            settings=reconstruction_settings,
            initial_microstructure=initial_microstructure)
    return characterization, convergence_data, last_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match a given microstructure by characterising and then reconstructing it.')
    parser.add_argument('--microstructure_filename', type=str, help='File of MS that shall be matched',
                        default='../example_microstructures/pymks_ms_64x64_2.npy')
    parser.add_argument('--initial_microstructure_file', type=str, help='Microstructure to initializa with. If None, random initialization is used', default=None)
    parser.add_argument('--data_folder', type=str, help='Results folder. If None, default with timestamp.', default='results')
    parser.add_argument('--descriptor_types', nargs='+', type=str, help='Descriptor types (list)', default=['Correlations'], choices=descriptor_choices)
    parser.add_argument('--descriptor_weights', nargs='+', type=float, help='Descriptor weights (list)', default=None)
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
    parser.add_argument('--add_dimension', type=int, help='Add dimension. Value in pixels or None', default=None)
    parser.add_argument('--tolerance', type=float, help='Resonctruction tolerance.', default=1e-10)
    parser.add_argument('--oor_multiplier', type=float, help='Penalty weight for OOR in loss', default=1000.0)
    parser.add_argument('--phase_sum_multiplier', type=float, help='Penalty weight for phase sum in loss', default=1000.0)
    parser.add_argument('--slice_mode', type=str, help='Average or sample slices?', default='average')
    parser.add_argument('--precision', type=int, help='Precision. 64 recommended', default=64)
    parser.add_argument('--information', type=str, help='Information that is added to files that are written.', default=None)
    parser.add_argument('--stride', type=int, help='Stride (3D DiffMCR).', default=4)
    parser.add_argument('--logfile_name', type=str, help='Name of logfile w/o extension.', default='logfile')
    parser.add_argument('--logging_level', type=int, help='Logging level.', default=logging.INFO)
    parser.add_argument('--logfile_date', dest='logfile_date', action='store_true')
    parser.set_defaults(logfile_date=False)
    parser.add_argument('--grey_values', dest='grey_values', action='store_true')
    parser.set_defaults(grey_values=False)
    parser.add_argument('--isotropic', dest='isotropic', action='store_true')
    parser.set_defaults(isotropic=False)
    parser.add_argument('--no_multiphase', dest='use_multiphase', action='store_false')
    parser.set_defaults(use_multiphase=True)
    parser.add_argument('--no_multigrid_descriptor', dest='use_multigrid_descriptor', action='store_false')
    parser.set_defaults(use_multigrid_descriptor=True)
    parser.add_argument('--use_multigrid_reconstruction', dest='use_multigrid_reconstruction', action='store_true')
    parser.set_defaults(use_multigrid_reconstruction=False)
    args = parser.parse_args()
    main(args)
