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
import argparse
import os
import pickle
import json
from typing import Dict, List, Tuple, Union

import numpy as np

def read_characterization(filename):
    """Read a microstructure characterization file."""
    with open(filename, 'rb') as f:
        ms_char = pickle.load(f)
    return ms_char

def merge(ms_chars: List[Dict[str, Union[np.ndarray, Tuple[np.ndarray]]]]):
    """Merge a list of three microstructure characterizations to a single characterization file where the descriptors
    from the given characterizations are interpreted as descriptors in a certain dimension only. The first descriptor
    is interpreted as descriptor for slices in the x-plane, second for y and third for z. When providing three 
    characterizations, it is important to note that the orientation must match, otherwise the descriptor is not realizable,
    i.e. it is not possible to find a microstructure that has this descriptor. For this reason, if the microstructure
    is only elongated / anisotropic in one direction, a list of length 2 can also be passed. In tht case, the elongated
    view on the data should be passed as fist argument of the list"""
    if len(ms_chars) == 1:
        char_1 = ms_chars[0]
        ms_chars = [char_1, char_1, char_1]
    if len(ms_chars) == 2:
        char_1, char_2 = ms_chars
        ms_chars = [char_1, char_1, char_2]
    if len(ms_chars) != 3:
        raise ValueError('Provided {len(ms_chars)} characterizations, but 2 or 3 are needed.')
    keys = ms_chars[0].keys()
    for ms_char in ms_chars:
        if keys != ms_char.keys():
            raise ValueError('The characterizations cannot be merged because they do not ' +
                    'contain the same descriptors.')

    merged_chars = {key: tuple([ms_char[key] for ms_char in ms_chars]) for key in keys}
    return merged_chars

def main(args):
    """Main function for merge script. This wraps some i/o around the merge() function in order to make it usable
    via a command line interface and via a GUI. When mcrpy is used as a Python module directly, this main function is not called."""
    char_files = [args.x_file, args.y_file, args.z_file] if args.z_file is not None else [args.x_file, args.y_file]
    for char_file in char_files:
        if not os.path.isfile(char_file):
            raise ValueError(f'Given file {char_file} does not exist!')
        if not char_file.endswith('.pickle'):
            raise ValueError(f'Given file {char_file} should end with .pickle!')

    if os.path.isfile(args.outfile):
        raise ValueError(f'Output file {args,outfile} does exist!')
    if not args.outfile.endswith('.pickle'):
        raise ValueError(f'Output file {args.outfile} should end with .pickle!')
    
    ms_chars = [read_characterization(char_file) for char_file in char_files]
    merged_chars = merge(ms_chars)
    with open(args.outfile, 'wb') as f:
        pickle.dump(merged_chars, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge multiple 2D chars to a single aniso 3D one for reconstruction")
    parser.add_argument('x_file')
    parser.add_argument('y_file')
    parser.add_argument('--z_file', default=None)
    parser.add_argument('outfile')
    args = parser.parse_args()
    main(args)

