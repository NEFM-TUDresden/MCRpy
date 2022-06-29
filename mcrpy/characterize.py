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
import logging
import pickle
import os
import tensorflow as tf
try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except:
    pass
import numpy as np
from typing import Any, Dict, List, Tuple, Union

from mcrpy.src import descriptor_factory
from mcrpy.src import fileutils
from mcrpy.src import log
from mcrpy.src import loader
from mcrpy.src import numerical_precision
from mcrpy.src.Settings import CharacterizationSettings
from mcrpy.src.descriptor_factory import descriptor_choices

def convert(x: Any) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """Convert x to numpy for serialization. If x is a np.ndarray, do nothing. If x is a tf.Tensor, convert it to numpy.
    If x is a list or a tuple, return a tuple where convert has been called on each element."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, tf.Tensor):
        return x.numpy()
    # multilevel recursion not intended 
    # but you are welcome to shoot yourself in the knee if you wish
    if isinstance(x, tuple) or isinstance(x, list):
        return tuple([convert(x_i) for x_i in x])
    raise NotImplementedError('Conversion to storable type not implemented')

@log.log_this
def make_descriptor_functions(
        microstructure_shape: Tuple[int], 
        settings: CharacterizationSettings
        ) -> Dict[str, callable]:
    """Create descriptor functions given the microstructure shape and some settings."""
    raw_descriptors = {descriptor_type: descriptor_factory.create(
        descriptor_type, arguments=vars(settings)) 
        for descriptor_type in settings.descriptor_types}
    if len(microstructure_shape) == 2:
        descriptors = raw_descriptors
    elif len(microstructure_shape) == 3:
        descriptors = {descriptor_type: descriptor_factory.permute(
            raw_descriptor, microstructure_shape, settings.np_dtype, 
            settings.tf_dtype, settings.n_phases, isotropic=settings.isotropic, mode=settings.slice_mode) 
            for descriptor_type, raw_descriptor in raw_descriptors.items()}
    else:
        raise ValueError('Microstructure must be 2D or 3D')
    return descriptors

@log.log_this
def run_characterization(x: tf.Tensor, descriptors: Dict[str, callable]) -> Dict[str, np.ndarray]:
    """Run the characterization for all descriptors."""
    return {descriptor_type: convert(descriptor_function(x)) 
            for descriptor_type, descriptor_function in descriptors.items()}

def main(args):
    """Main function for characterization script. This wraps some i/o around the characterize() function in order to make it usable
    via a command line interface and via a GUI. When mcrpy is used as a Python module directly, this main function is not called."""
    settings = CharacterizationSettings(**vars(args))

    # set up files, folders and logging
    settings.target_folder = fileutils.create_target_folder(settings)
    microstructure_filename = fileutils.copy_ms_to_target(settings.target_folder, settings)
    microstructure = fileutils.read_ms(microstructure_filename)

    # run characterization
    characterization = characterize(microstructure, settings)

    # save
    info_file_additive = '.'.join(
            os.path.split(microstructure_filename)[-1].split('.')[:-1]) + '_' + (
                    settings.information + '_' if settings.information is not None else '')
    save_characterization(characterization, info_file_additive, settings.target_folder)

def save_characterization(
        characterization: Dict[str, Union[np.ndarray, Tuple[np.ndarray]]],
        info_file_additive: str,
        target_folder: str):
    """Save characterization to a pickle file."""
    char_file = os.path.join(target_folder, '{}characterization.pickle'.format(
        info_file_additive))
    with open(char_file, 'wb') as f:
        pickle.dump(characterization, f)
    logging.info(f'Saved descriptor to {char_file}')

def characterize(
        microstructure: np.ndarray, 
        settings: CharacterizationSettings = None
        ) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray]]]:
    """Characterize a microstructure using statistical descriptors. The types of descriptors 
    and various settings can be controlled using a CharacterizationSettings object. For an 
    overview of all possible settings, see (a) the definition if CharacterizationSettings in
    src.Settings, (b) the --help in the command line interface or (c) the GUI. For a description 
    of all possible descriptors see mcrpy.descriptors and the respective documentations."""
    microstructure = microstructure.astype(np.int)
    if settings is None:
        settings = CharacterizationSettings()

    # check MS format
    if 1 in microstructure.shape:
        raise ValueError(f'Provided microstructure np.ndarray has shape ' + 
            f'{microstructure.shape}, which contains ones. The microstructure shape should ' + 
            'not contain ones.')
    if len(microstructure.shape) not in {2, 3}:
        raise ValueError(f'Provided microstructure np.ndarray has shape {microstructure.shape} ' +
            f'and is therefore {len(microstructure.shape)}-dimensional, but it should be 2- ' +
            f'or 3-dimensional.')

    # setup logging
    log.setup_logging(settings.target_folder, settings)

    # encode microstructure
    encoded_ms = fileutils.encode_ms(microstructure, use_multiphase=settings.use_multiphase, grey_values=settings.grey_values)
    extended_shape = (1, *encoded_ms.shape)

    # augment settings
    settings.desired_shape_2d = tuple(microstructure.shape[:2])
    settings.desired_shape_extended = extended_shape
    settings.n_phases = extended_shape[-1]
    numerical_precision.set_precision(settings, settings.precision)

    # make descriptor functions
    loader.load_plugins([f'mcrpy.descriptors.{descriptor_type}' 
        for descriptor_type in settings.descriptor_types])
    descriptors = make_descriptor_functions(microstructure.shape, settings)

    # run characterization
    x = tf.Variable(encoded_ms.reshape(extended_shape), trainable=False,
            name='ms', dtype=settings.tf_dtype)
    characterization = run_characterization(x, descriptors)
    return characterization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Characteriza a microstructure given as numpy array')
    parser.add_argument('microstructure_filename', type=str, help='File of MS to characterize',
                        default='../example_microstructures/pymks_ms_64x64_2.npy')
    parser.add_argument('--data_folder', type=str, help='Results folder. If None, default with timestamp.', default='results')
    parser.add_argument('--descriptor_types', nargs='+', type=str, help='Descriptor types (list)', default=['Correlations'], choices=descriptor_choices)
    parser.add_argument('--nl_method', type=str, help='Nonlinearity method.', default='relu')
    parser.add_argument('--limit_to', type=int, help='Limit in pixels to which to limit the characterization metrics.', default=8)
    parser.add_argument('--threshold_steepness', type=float, help='Steepness of soft threshold function. Regularisation parameter.', default=10)
    parser.add_argument('--gram_weights_filename', type=str, help='Gram weigths filename wo path.', default='vgg19_normalized.pkl')
    parser.add_argument('--slice_mode', type=str, help='Average or sample slices?', default='average')
    parser.add_argument('--precision', type=int, help='Precision. 64 recommended', default=64)
    parser.add_argument('--information', type=str, help='Information that is added to files that are written.', default=None)
    parser.add_argument('--logfile_name', type=str, help='Name of logfile w/o extension.', default='logfile')
    parser.add_argument('--logging_level', type=int, help='Logging level.', default=logging.INFO)
    parser.add_argument('--grey_values', dest='grey_values', action='store_true')
    parser.set_defaults(grey_values=False)
    parser.add_argument('--isotropic', dest='isotropic', action='store_true')
    parser.set_defaults(isotropic=False)
    parser.add_argument('--logfile_date', dest='logfile_date', action='store_true')
    parser.set_defaults(logfile_date=False)
    parser.add_argument('--no_multiphase', dest='use_multiphase', action='store_false')
    parser.set_defaults(use_multiphase=True)
    parser.add_argument('--no_multigrid_descriptor', dest='use_multigrid_descriptor', action='store_false')
    parser.set_defaults(use_multigrid_descriptor=True)
    args = parser.parse_args()
    main(args)
