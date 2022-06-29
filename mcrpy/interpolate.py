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
import json
import logging
import os
import pickle
import time

from typing import Dict, List

import numpy as np

from mcrpy.src import log
from mcrpy.src import fileutils

@log.log_this
def read_descriptors(args: argparse.Namespace) -> List[Dict[str, np.ndarray]]:
    """Read the descriptors and return them as a list."""
    ms_chars = []
    microstructure_descriptors = [args.microstructure_descriptor_1, args.microstructure_descriptor_2]
    for microstructure_descriptor in microstructure_descriptors:
        with open(microstructure_descriptor, 'rb') as f:
            ms_char = pickle.load(f)
        ms_chars.append(ms_char)
    keys = ms_chars[0].keys()
    for ms_char in ms_chars:
        if not keys == ms_char.keys():
            raise ValueError('Key Mismatch! Make sure all characterization files '+
                    'contain the same descriptors.')
    return ms_chars

@log.log_this
def interpolate(
        ms_char_from: Dict[str, np.ndarray], 
        ms_char_to: Dict[str, np.ndarray], 
        n_steps: int):
    """Interpolate between two diferent microstructure characterizations in the descriptor space."""
    if n_steps <= 2:
        raise ValueError(f'Number of steps is {n_steps}, but should be > 2 because steps ' + 
                f'include start and end.')
    interpolated_descriptors = []
    for i in range(n_steps):
        d_inter = {}
        for key in ms_char_from:
            d_from = ms_char_from[key]
            d_to = ms_char_to[key]
            merged = isinstance(d_from, tuple)
            if merged:
                ddi_l = []
                for ii in range(len(d_from)):
                    og_shape = d_from[ii].shape
                    if not d_to[ii].shape == og_shape:
                        raise ValueError('Shape mismatch! Make sure the descriptors to ' +
                                'interpolate have the same shape.')
                    ddl = d_from[ii].flatten()
                    ddr = d_to[ii].flatten()
                    ddd = ddr - ddl
                    ddii = (ddl + ddd / (n_steps - 1) * i).reshape(og_shape)
                    ddi_l.append(ddii)
                ddi = tuple(ddi_l)
            else:
                og_shape = d_from.shape
                if not d_to.shape == og_shape:
                    raise ValueError('Shape mismatch! Make sure the descriptors to ' +
                            'interpolate have the same shape.')
                ddl = d_from.flatten()
                ddr = d_to.flatten()
                ddd = ddr - ddl
                ddi = (ddl + ddd / (n_steps - 1) * i).reshape(og_shape)
            d_inter[key] = ddi
        interpolated_descriptors.append(d_inter)
    return interpolated_descriptors


def main(args):
    """Main function for interpolation script. This wraps some i/o around the interpolate() function in order to make it usable
    via a command line interface and via a GUI. When mcrpy is used as a Python module directly, this main function is not called."""
    if not os.path.isfile(args.microstructure_descriptor_1):
        raise ValueError(f'Given file {args.microstructure_descriptor_1} does not exist!')
    if not args.microstructure_descriptor_1.endswith('.pickle'):
        raise ValueError(f'Given file {args.microstructure_descriptor_1} should end with .pickle!')
    if not os.path.isfile(args.microstructure_descriptor_2):
        raise ValueError(f'Given file {args.microstructure_descriptor_2} does not exist!')
    if not args.microstructure_descriptor_2.endswith('.pickle'):
        raise ValueError(f'Given file {args.microstructure_descriptor_2} should end with .pickle!')

    # setup files, folders and logging
    target_folder = fileutils.create_target_folder(args)
    log.setup_logging(target_folder, args)

    # read descriptors
    ms_char_from, ms_char_to = read_descriptors(args)

    # interpolate
    interpolated_descriptors = interpolate(ms_char_from, ms_char_to, args.n_steps)

    # save interpolated descriptors
    for i, interpolated_descriptor in enumerate(interpolated_descriptors):
        filename = os.path.join(target_folder, f'des_interpolated_{i}.pickle')
        with open(filename, 'wb') as f:
            pickle.dump(interpolated_descriptor, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpolate in descriptor space')
    parser.add_argument('microstructure_descriptor_1', type=str, help='MS 1')
    parser.add_argument('microstructure_descriptor_2', type=str, help='MS 2')
    parser.add_argument('--n_steps', type=int, help='Number of interpolated MSs', default=3)
    parser.add_argument('--data_folder', type=str, help='Folder that results shall be written to', default=None)
    parser.add_argument('--logfile_name', type=str, default='logfile')
    parser.add_argument('--logging_level', type=int, default=logging.INFO)
    parser.add_argument('--logfile_date', dest='logfile_date', action='store_true')
    parser.set_defaults(logfile_date=False)
    args = parser.parse_args()
    main(args)

