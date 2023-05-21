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
import logging
import pickle
import os
from typing import Any, Callable, Dict, List, Tuple, Union
import tensorflow as tf
with contextlib.suppress(Exception):
    tf.config.experimental.enable_tensor_float_32_execution(False)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

from mcrpy.src import descriptor_factory
from mcrpy.src import fileutils
from mcrpy.src import log
from mcrpy.src import loader
from mcrpy.src.Microstructure import Microstructure
from mcrpy.src.Settings import CharacterizationSettings
from mcrpy.src.Symmetry import Symmetry, symmetries
from mcrpy.src.descriptor_factory import descriptor_choices, descriptor_classes

def convert(x: Any) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """Convert x to numpy for serialization. If x is a np.ndarray, do nothing. If x is a tf.Tensor, convert it to numpy.
    If x is a list or a tuple, return a tuple where convert has been called on each element."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, tf.Tensor):
        return x.numpy()
    # multilevel recursion not intended 
    # but you are welcome to shoot yourself in the knee if you wish
    if isinstance(x, (tuple, list)):
        return tuple(convert(x_i) for x_i in x)
    raise NotImplementedError('Conversion to storable type not implemented')

@log.log_this
def make_descriptor_functions(
        microstructure_shape: Tuple[int], 
        settings: CharacterizationSettings
        ) -> Dict[str, callable]:
    """Create descriptor functions given the microstructure shape and some settings."""
    if len(microstructure_shape) == 2:
        raw_descriptors = {descriptor_type: descriptor_factory.create(
            descriptor_type, arguments=vars(settings)) 
            for descriptor_type in settings.descriptor_types}
        def apply_descriptor(d: Callable):
            def inner(ms: Microstructure):
                return d(ms.xx)
            return inner
        descriptors = {descriptor_type: apply_descriptor(raw_descriptor)
            for descriptor_type, raw_descriptor in raw_descriptors.items()}
    elif len(microstructure_shape) == 3:
        descriptors = {descriptor_type: descriptor_factory.permute(
            descriptor_type, microstructure_shape, settings.n_phases, 
            isotropic=settings.isotropic, mode=settings.slice_mode,
            arguments=vars(settings)) 
            for descriptor_type in settings.descriptor_types}
    else:
        raise ValueError('Microstructure must be 2D or 3D')
    return descriptors

@log.log_this
def run_characterization(
        ms: Microstructure, 
        descriptors: Dict[str, callable]) -> Dict[str, np.ndarray]:
    """Run the characterization for all descriptors."""
    return {descriptor_type: convert(descriptor_function(ms)) 
            for descriptor_type, descriptor_function in descriptors.items()}

def main(args):
    """Main function for characterization script. This wraps some i/o around the characterize() function in order to make it usable
    via a command line interface and via a GUI. When mcrpy is used as a Python module directly, this main function is not called."""
    settings = CharacterizationSettings(**vars(args))

    # set up files, folders and logging
    if settings.microstructure_filenames is None:
        raise ValueError('Give at least one microstructure_filename to characterize.')
    settings.target_folder = fileutils.create_target_folder(settings)
    microstructure_filenames = fileutils.copy_ms_to_target(settings.target_folder, settings)
    microstructures = [Microstructure.from_npy(microstructure_filename, use_multiphase=settings.use_multiphase, trainable=False)
            for microstructure_filename in microstructure_filenames]

    # run characterization
    characterizations = characterize(microstructures, settings)

    # make sure to use a list of characterizations
    if not isinstance(characterizations, list):
        characterizations = [characterizations]

    # save
    for characterization, microstructure_filename in zip(characterizations, microstructure_filenames):
        info_file_additive = (
            '.'.join(
                os.path.split(microstructure_filename)[-1].split('.')[:-1]
            )
            + '_'
            + (
                f'{settings.information}_'
                if settings.information is not None
                else ''
            )
        )
        save_characterization(characterization, info_file_additive, settings.target_folder)

def save_characterization(
        characterization: Dict[str, Union[np.ndarray, Tuple[np.ndarray]]],
        info_file_additive: str,
        target_folder: str):
    """Save characterization to a pickle file."""
    char_file = os.path.join(
        target_folder, f'{info_file_additive}characterization.pickle'
    )
    with open(char_file, 'wb') as f:
        pickle.dump(characterization, f)
    logging.info(f'Saved descriptor to {char_file}')

def characterize(
        microstructures: Union[Microstructure, List[Microstructure]], 
        settings: CharacterizationSettings = None
        ) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray]]]:
    """Characterize microstructure(s) using statistical descriptors. The types of descriptors 
    and various settings can be controlled using a CharacterizationSettings object. For an 
    overview of all possible settings, see (a) the definition if CharacterizationSettings in
    src.Settings, (b) the --help in the command line interface or (c) the GUI. For a description 
    of all possible descriptors see mcrpy.descriptors and the respective documentations."""
    if settings is None:
        settings = CharacterizationSettings()

    # setup logging
    log.setup_logging(settings.target_folder, settings)

    # make sure microstructures are a list
    if isinstance(microstructures, Microstructure):
        microstructures = [microstructures]

    # assert that microstructures are in same format
    if (
        len({microstructure.shape for microstructure in microstructures}) != 1
        or len({microstructure.n_phases for microstructure in microstructures})
        != 1
        or len(
            {microstructure.has_phases for microstructure in microstructures}
        )
        != 1
    ):
        raise ValueError("""The microstructures to characterize within one call to the function
                mcrpy.characterize must have the same resolution, number of phases and should
                either all contain orientations or none of them should. This is because the
                descriptor function is created once and evaluated on all structures. If you want
                to characterize such different microstructures, consider splitting up the 
                characterization intp multiple calls to mcrpy.characterize""")

    # augment settings
    settings.desired_shape_2d = tuple(microstructures[0].shape[:2])
    settings.desired_shape_extended = microstructures[0].x_shape
    settings.n_phases = microstructures[0].n_phases
    if settings.symmetry is not None and not isinstance(
        settings.symmetry, Symmetry
    ):
        assert settings.symmetry in symmetries
        settings.symmetry = symmetries[settings.symmetry]
    tf.keras.backend.set_floatx("float64")

    # load modules
    loader.load_plugins([f'mcrpy.descriptors.{descriptor_type}' 
        for descriptor_type in settings.descriptor_types])

    # make descriptor functions
    descriptors = make_descriptor_functions(microstructures[0].spatial_shape, settings)

    # run characterization
    characterizations = [run_characterization(microstructure, descriptors) for microstructure in microstructures]

    # add settings to characterization as metadata
    for characterization in characterizations:
        characterization['settings'] = settings

    # extract lone characterization if only one microstructure was given
    if len(characterizations) == 1:
        characterizations = characterizations[0]
    return characterizations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Characterize a microstructure given as numpy array')
    parser.add_argument('--microstructure_filenames', nargs='+', type=str, help='File or files of MS to characterize', default=None)
    parser.add_argument('--data_folder', type=str, help='Results folder. If None, default with timestamp.', default='results')
    parser.add_argument('--descriptor_types', nargs='+', type=str, help='Descriptor types (list)', default=['Correlations'], choices=descriptor_choices)
    parser.add_argument('--nl_method', type=str, help='Nonlinearity method.', default='relu')
    parser.add_argument('--limit_to', type=int, help='Limit in pixels to which to limit the characterization metrics.', default=8)
    parser.add_argument('--threshold_steepness', type=float, help='Steepness of soft threshold function. Regularisation parameter.', default=10)
    parser.add_argument('--gram_weights_filename', type=str, help='Gram weigths filename wo path.', default='vgg19_normalized.pkl')
    parser.add_argument('--slice_mode', type=str, help='Average or sample slices?', default='average')
    parser.add_argument('--information', type=str, help='Information that is added to files that are written.', default=None)
    parser.add_argument('--symmetry', type=str, help='Symmetry of the microstructure if orientations are considered. Default is None.', default=None)
    parser.add_argument('--logfile_name', type=str, help='Name of logfile w/o extension.', default='logfile')
    parser.add_argument('--logging_level', type=int, help='Logging level.', default=logging.INFO)
    parser.add_argument('--non_periodic', dest='periodic', action='store_false')
    parser.set_defaults(periodic=True)
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
