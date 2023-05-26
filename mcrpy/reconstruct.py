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
import contextlib
import pickle
import os
import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import tensorflow as tf
with contextlib.suppress(Exception):
    tf.config.experimental.enable_tensor_float_32_execution(False)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from mcrpy.src import log
from mcrpy.src import fileutils
from mcrpy.src import descriptor_factory
from mcrpy.src import profile
from mcrpy.src import DMCR
from mcrpy.src.Settings import ReconstructionSettings
from mcrpy.src.Symmetry import Symmetry, symmetries
from mcrpy.src.Microstructure import Microstructure
from mcrpy.src.descriptor_factory import descriptor_choices


def main(args):
    """Main function for reconstruction script. This wraps some i/o around the reconstruct() function in order to make it usable
    via a command line interface and via a GUI. When mcrpy is used as a Python module directly, this main function is not called."""
    # prepare reconstruction
    descriptor_dict, desired_shape, initial_microstructure, settings = prepare_reconstruction(args)

    # reconstruct
    convergence_data, last_frame = reconstruct(
            descriptor_dict, 
            desired_shape, 
            settings=settings,
            initial_microstructure=initial_microstructure)

    # save results
    save_results(settings, convergence_data, last_frame)

def save_results(settings, convergence_data, last_frame):
    information_additives = (
        '' if settings.information is None else f'_{settings.information}'
    )
    foldername = settings.target_folder if settings.target_folder is not None else ''

    # add settings to convergence data
    convergence_data['settings'] = settings

    # save convergence data
    save_convergence_data(convergence_data, foldername, information_additives=information_additives)

    # save last state separately
    save_last_state(last_frame, information_additives, foldername)

def save_last_state(last_frame, information_additives, foldername):
    np_data_filename = os.path.join(
        foldername, f'last_frame{information_additives}.npy'
    )
    last_frame.to_npy(np_data_filename)

def prepare_reconstruction(args):
    args.target_folder = fileutils.create_target_folder(args)
    descriptor_dict = fileutils.load(args.descriptor_filename)
    desired_shape = (args.extent_x, args.extent_y, args.extent_z) if args.extent_z not in {None, 0, 1} else (args.extent_x, args.extent_y)
    del args.extent_x
    del args.extent_y
    del args.extent_z
    initial_microstructure = None if args.initial_microstructure_file is None \
            else Microstructure.from_npy(args.initial_microstructure_file, use_multiphase=args.use_multiphase)
    del args.initial_microstructure_file
    settings = ReconstructionSettings(**vars(args))
    return descriptor_dict,desired_shape,initial_microstructure,settings


def save_convergence_data(convergence_data: Dict[str, Any], foldername: str, information_additives: str = ''):
    filename = f'convergence_data{information_additives}.pickle'
    convergence_data_filename = os.path.join(foldername, filename)
    with open(convergence_data_filename, 'wb') as f:
        pickle.dump(convergence_data, f, protocol=4)

def reconstruct(
        descriptor_dict: Dict[str, Union[np.ndarray, Tuple[np.ndarray]]],
        desired_shape: Tuple[int],
        settings: ReconstructionSettings = None,
        initial_microstructure: Microstructure = None):
    settings, desired_descriptors = setup_reconstruction(descriptor_dict, desired_shape, settings, initial_microstructure)

    # run reconstruction
    diff_mcr = DMCR.DMCR(**vars(settings))
    convergence_data, last_frame = diff_mcr.reconstruct(desired_descriptors, desired_shape)
    del diff_mcr
    return convergence_data, last_frame

def setup_reconstruction(descriptor_dict, desired_shape, settings, initial_microstructure):
    settings = setup_settings(descriptor_dict, desired_shape, settings, initial_microstructure)

    # set up files, folders and logging
    log.setup_logging(settings.target_folder, settings)
    settings.save_to = settings.target_folder 

    # get characterization settings
    characterization_settings = descriptor_dict.pop('settings')

    # flatten and weight the descriptor
    desired_descriptors = [descriptor_dict[d] for d in settings.descriptor_types]

    # augment settings
    if 'VolumeFractions' in descriptor_dict:
        settings.volume_fractions = descriptor_dict['VolumeFractions']
    settings.n_phases = characterization_settings.n_phases
    
    # copy settings
    profile.PROFILE = settings.profile
    return settings, desired_descriptors

def setup_settings(descriptor_dict, desired_shape, settings, initial_microstructure):
    if settings is None:
        settings = ReconstructionSettings()
    if any(s_i in {0, 1, None} for s_i in desired_shape):
        raise ValueError(f'Provided desired_shape {desired_shape} should not contain ' +
            'entries that are 0, 1 or None.')
    if len(desired_shape) not in {2, 3}:
        raise ValueError(
            (
                f'Provided desired shape {desired_shape} has '
                + f'{len(desired_shape)} entries for {len(desired_shape)} microstructure '
                + 'dimensions, but only 2- and 3-dimensional microstructures are supported.'
            )
        )
    if initial_microstructure is not None:
        assert isinstance(initial_microstructure, Microstructure)
        settings.initial_microstructure = initial_microstructure
    if settings.use_multigrid_reconstruction and not settings.use_multigrid_descriptor:
        raise ValueError("""Provided settings.use_multigrid_reconstruction is True, 
                but settings.use_multigrid_descriptor is False. Cannot do 
                multigrid reconstruction withour mulrigrid descriptors, 
                although multigrid descriptors withour multigrid reconstruction is 
                possible and can be reasonable. Please fix the settings. Also, 
                if you want multigrid reconstruction, make sure the descriptor you 
                start with is a multigrid descriptor.""")
    if settings.grey_values and settings.use_multiphase:
        raise ValueError('Cannot set both grey_values and use_multiphase!')
    if settings.symmetry is not None and not isinstance(
        settings.symmetry, Symmetry
    ):
        assert settings.symmetry in symmetries
        settings.symmetry = symmetries[settings.symmetry]
    if 'settings' not in descriptor_dict.keys():
        raise ValueError("""The descriptor_dict passed to mcrpy.reconstruct must
                contain the mcrpy.CharacterizationSettings under the key settings.
                Characterizations carried out with previous versions might not contain
                this information. In this case, please re-characterize the structures
                with the current version of mcrpy or add the information manually.""")
                
    return settings


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
    parser.add_argument('--initial_temperature', type=float, help='Initial temperature for annealing', default=0.0001)
    parser.add_argument('--final_temperature', type=float, help='Final temperature for annealing', default=None)
    parser.add_argument('--cooldown_factor', type=float, help='Cooldown factor for annealing', default=0.9)
    parser.add_argument('--mutation_rule', type=str, help='Mutation rule for YT', default='relaxed_neighbor')
    parser.add_argument('--acceptance_distribution', type=str, help='Acceptance distribution for YT', default='zero_tolerance')
    parser.add_argument('--symmetry', type=str, help='Symmetry of the microstructure if orientations are considered. Default is None.', default=None)
    parser.add_argument('--max_iter', type=int, help='Maximum number of iterations.', default=500)
    parser.add_argument('--convergence_data_steps', type=int, help='Each x steps write data', default=10)
    parser.add_argument('--outfile_data_steps', type=int, help='Each x steps write data to disk', default=None)
    parser.add_argument('--tolerance', type=float, help='Resonctruction tolerance.', default=1e-10)
    parser.add_argument('--oor_multiplier', type=float, help='Penalty weight for OOR in loss', default=1000.0)
    parser.add_argument('--phase_sum_multiplier', type=float, help='Penalty weight for phase sum in loss', default=1000.0)
    parser.add_argument('--information', type=str, help='Information that is added to files that are written.', default=None)
    parser.add_argument('--logfile_name', type=str, help='Name of logfile w/o extension.', default='logfile')
    parser.add_argument('--logging_level', type=int, help='Logging level.', default=logging.INFO)
    parser.add_argument('--logfile_date', dest='logfile_date', action='store_true')
    parser.set_defaults(logfile_date=False)
    parser.add_argument('--non_periodic', dest='periodic', action='store_false')
    parser.set_defaults(periodic=True)
    parser.add_argument('--grey_values', dest='grey_values', action='store_true')
    parser.set_defaults(grey_values=False)
    parser.add_argument('--use_multiphase', dest='use_multiphase', action='store_true')
    parser.set_defaults(use_multiphase=False)
    parser.add_argument('--no_multigrid_descriptor', dest='use_multigrid_descriptor', action='store_false')
    parser.set_defaults(use_multigrid_descriptor=True)
    parser.add_argument('--use_multigrid_reconstruction', dest='use_multigrid_reconstruction', action='store_true')
    parser.set_defaults(use_multigrid_reconstruction=False)
    parser.add_argument('--greedy', dest='greedy', action='store_true')
    parser.set_defaults(greedy=False)
    parser.add_argument('--profile', dest='profile', action='store_true')
    parser.set_defaults(profile=False)
    parser.add_argument('--batch_size', type=float, help='Batch size for greedy optimization in 3D, ie how many slices are considered per step', default=1)
    args = parser.parse_args()
    main(args)
