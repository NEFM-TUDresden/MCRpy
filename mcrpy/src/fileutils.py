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
import os
import pickle
import shutil
import time
import subprocess
from typing import List, Dict, Union

import numpy as np
from mcrpy.src.Microstructure import Microstructure


def expand_folder_name(folder_name: str, intermediate_folders: List[str] = None) -> str:
    """Expand folder name by adding the current working directory and the intermediate folders.
    If the current working directory is already given in folder_name, then nothing is done.
    Examples:
        If the current working directory is /home/username/Dokumente/PSPLinkages/structure-characterisation,
        then expand_folder_name('TestFolder')
        returns /home/username/Dokumente/PSPLinkages/structure-characterisation/TestFolder
        and expand_folder_name('TestFolder', intermediate_folders=['TiAl', 'Synthetic'])
        returns /home/username/Dokumente/PSPLinkages/structure-characterisation/TiAl/Synthetic/TestFolder
    Handles ../ in fildername, unlike os.path.abspath """
    if intermediate_folders is None:
        intermediate_folders = []
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
        args.data_folder = f'DefaultFolder {time.asctime()}'.replace(
            ' ', '-'
        ).replace(':', '-')
    target_folder = expand_folder_name(args.data_folder)
    os.makedirs(target_folder, exist_ok=True)
    return target_folder


def copy_ms_to_target(target_folder: str, args: argparse.Namespace) -> str:
    """Copy microstructure to target folder."""
    microstructures_copyto = []
    for microstructure_filename in args.microstructure_filenames:
        microstructure_copyto = os.path.join(
            target_folder, os.path.split(microstructure_filename)[1])
        with contextlib.suppress(shutil.SameFileError):
            shutil.copy(microstructure_filename, microstructure_copyto)
        microstructures_copyto.append(microstructure_copyto)
    return microstructures_copyto

def load(filename: str, use_multiphase: bool = False) -> Union[np.ndarray, Dict]:
    if filename.endswith('.npy'):
        return Microstructure.from_npy(filename, use_multiphase=use_multiphase)
    elif filename.endswith('.pickle'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        raise ValueError('Can only load microstructures as npy- or pickle-files and descriptors as pickle-files. ' +
                f'The given filename {filename} is neither.')
