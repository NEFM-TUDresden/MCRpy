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
import os
import pickle
import shutil
import time
import subprocess
from typing import List, Dict, Union

import numpy as np


def expand_folder_name(folder_name: str, intermediate_folders: List[str] = []) -> str:
    """Expand folder name by adding the current working directory and the intermediate folders.
    If the current working directory is already given in folder_name, then nothing is done.
    Examples:
        If the current working directory is /home/username/Dokumente/PSPLinkages/structure-characterisation,
        then expand_folder_name('TestFolder')
        returns /home/username/Dokumente/PSPLinkages/structure-characterisation/TestFolder
        and expand_folder_name('TestFolder', intermediate_folders=['TiAl', 'Synthetic'])
        returns /home/username/Dokumente/PSPLinkages/structure-characterisation/TiAl/Synthetic/TestFolder
    Handles ../ in fildername, unlike os.path.abspath """
    base_folder = os.getcwd()
    if base_folder in folder_name:
        return folder_name
    while folder_name.startswith('../'):
        folder_name = folder_name[3:]
        base_folder = os.path.split(base_folder)[0]
    for intermediate_folder in intermediate_folders:
        base_folder = os.path.join(base_folder, intermediate_folder)
    return os.path.join(base_folder, folder_name)


def create_target_folder(args: argparse.Namespace) -> str:
    """Create target folder where all results are stored."""
    if args.data_folder is None:
        args.data_folder = 'DefaultFolder {}'.format(
            time.asctime()).replace(' ', '-').replace(':', '-')
    target_folder = expand_folder_name(args.data_folder)
    os.makedirs(target_folder, exist_ok=True)
    return target_folder


def copy_ms_to_target(target_folder: str, args: argparse.Namespace) -> str:
    """Copy microstructure to target folder."""
    microstructure_copyto = os.path.join(
        target_folder, os.path.split(args.microstructure_filename)[1])
    try:
        shutil.copy(args.microstructure_filename, microstructure_copyto)
    except shutil.SameFileError:
        pass
    return microstructure_copyto

def load(filename: str) -> Union[np.ndarray, Dict]:
    if filename.endswith('.npy'):
        return read_ms(filename)
    elif filename.endswith('.pickle'):
        return read_descriptor(descriptor_filename)
    else:
        raise ValueError('Can only load microstructures as npy-files and descriptors as pickle-files. ' +
                f'The given filename {filename} is neither.')


def read_ms(microstructure_filename: str) -> np.ndarray:
    """Return a microstructure given the filename."""
    microstructure = None
    if microstructure_filename.endswith('.npy'):
        ms_raw = np.load(microstructure_filename)
        if np.min(ms_raw) < -0.1:
            logging.warning(f'Assuming old decoding, adding 0.5 to MS')
            ms_raw += 0.5
        microstructure = np.round(ms_raw).astype(np.int8)
    elif microstructure_filename.endswith('.pickle'):
        import pickle
        with open(microstructure_filename, 'rb') as f:
            convergence_data = pickle.load(f)
        if 'og_img' not in convergence_data.keys():
            raise ValueError('There is no microstructure under the key ' + 
                    'og_ms in this pickle-file. Maybe you meant to open '+ 
                    'a npy-file?')
        microstructure = convergence_data['og_img']
    else:
        raise NotImplementedError
    return microstructure

def read_descriptor(descriptor_filename: str) -> Dict:
    with open(descriptor_filename, 'rb') as f:
        descriptor = pickle.load(f)
    return descriptor

def encode_ms(microstructure: np.ndarray, use_multiphase: bool = True, grey_values: bool = False) -> np.ndarray:
    if grey_values:
        microstructure = (microstructure.astype(np.float64) - np.min(microstructure)) / (np.max(microstructure) - np.min(microstructure))
    if use_multiphase and not grey_values:
        phases = np.unique(microstructure)
        logging.info(f'encoding phases {phases}')
        n_phases = len(phases)
        if not all(np.array([i for i in range(len(phases))]) == phases):
            raise ValueError('Phases should be numbered consecutively, starting at 0.')
        encoded_ms = np.zeros(tuple([*microstructure.shape, n_phases]), np.int8)
        for phase in phases:
            encoded_ms[..., phase] = microstructure == phase
        return encoded_ms
    else:
        return microstructure.reshape(tuple(
            [*microstructure.shape, 1])).astype(np.float32)

def decode_ms(microstructure: np.ndarray) -> np.ndarray:
    if microstructure.shape[0] == 1:
        microstructure = microstructure[0]
    n_phases = microstructure.shape[-1]
    ms_decoded_shape = microstructure.shape[:-1]
    if n_phases == 1:
        return np.round(microstructure).reshape(ms_decoded_shape)
    n_pixels = np.product(microstructure.shape[:-1])
    ms_reshaped = microstructure.reshape((n_pixels, -1))
    ms_decoded = np.zeros(n_pixels)
    for pixel in range(n_pixels):
        ms_decoded[pixel] = np.argmax(ms_reshaped[pixel])
    return ms_decoded.reshape(ms_decoded_shape)

if __name__ == "__main__":
    ms = np.array([[0, 2], [1, 1]], np.int8)
    print(encode_ms(ms))
