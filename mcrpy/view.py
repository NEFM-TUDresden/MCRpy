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

import os
import subprocess
import logging
import pickle
import argparse
from typing import Any, Dict, List, Union

import numpy as np

from mcrpy.src import descriptor_factory
from mcrpy.src import loader
from mcrpy.src import log
from mcrpy.src import fileutils
from mcrpy.src.Microstructure import Microstructure
from mcrpy.descriptors.Descriptor import Descriptor
from mcrpy.src.Symmetry import symmetries, Symmetry

@log.log_this
def view_generic_pickle(data: Dict, save_as: str = None, original_ms: Microstructure = None):
    if not isinstance(data, dict):
        raise NotImplementedError('Only pickles that contain dictionaries can be viewed.')
    if save_as is not None:
        assert save_as.endswith('.png')
    logging.info(f'Data contains keys {data.keys()}')
    if 'scatter_data' in data and 'raw_data' in data:
        view_convergence_data(data, original_ms)
    else:
        for k, v in data.items():
            if k == 'settings':
                continue
            # try:
            loader.load_plugins([f'mcrpy.descriptors.{k}' ])
            visualize = descriptor_factory.get_visualization(k)
            visualize(v, descriptor_type=k, 
                    save_as=f'{save_as[:-4]}_{k}.png' if save_as is not None else None)
            # except Exception as e:
            #     logging.error(f'Error plotting {k}')
        logging.info('done visualizations')

@log.log_this
def view_convergence_data(convergence_data: Dict[str, np.ndarray], original_ms: Microstructure = None):
    """View a convergence_data file."""
    from mcrpy.src.point_browser import PointBrowser

    scatter_data = convergence_data['scatter_data'].astype(np.float32)
    raw_data = convergence_data['raw_data']
    line_datas = {'Cost': convergence_data['line_data']}
    settings = convergence_data.get('settings', None)
    PointBrowser(scatter_data, raw_data, line_datas, original_ms=original_ms, log_axis=True, settings=settings)

@log.log_this
def view_microstructure(ms: Microstructure, save_as: str = None, axis: bool = True, cmap: str = 'cividis', symmetry: Symmetry = None):
    import matplotlib
    import matplotlib.pyplot as plt
    if ms.is_2D:
        import tensorflow as tf
        if ms.has_phases:
            x = ms.decode_phases()
        else:
            x = ms.ori.numpy()[0] if symmetry is None else symmetry.to_ipf(ms.ori)[0]
        # matplotlib.rcParams.update({
        #     "pgf.texsystem":"pdflatex",
        #     'font.family': 'serif',
        #     'font.size': 12,
        #     'figure.titlesize': 'medium',
        #     'text.usetex':'True',
        #     'pgf.rcfonts':'False',
        #     "pgf.preamble": r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{mathrsfs}"
        #     })
        if axis:
            plt.figure(figsize=(2.5, 2.5))
            if ms.has_phases:
                plt.imshow(x, cmap=cmap)
            else:
                plt.imshow(x)
            plt.title(r'$M_{ij}$')
            plt.xlabel(r'$i$ in Px')
            plt.ylabel(r'$j$ in Px')
            x_max, y_max = x.shape[0], x.shape[1]
            xticks = [0, x_max // 2 - 1, x_max - 1]
            yticks = [0, y_max // 2 - 1, y_max - 1]
            plt.gca().set_xticks(xticks)
            plt.gca().set_yticks(yticks)
            plt.gca().set_xticklabels([x_i + 1 for x_i in xticks])
            plt.gca().set_yticklabels(reversed([y_i + 1 for y_i in yticks]))
            plt.tight_layout()
        else:
            sizes = np.shape(ms)
            fig = plt.figure()
            fig.set_size_inches(1, 1, forward = False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            if ms.has_phases:
                ax.imshow(x, cmap=cmap)
            else:
                ax.imshow(x)
        if save_as:
            if axis:
                plt.savefig(save_as, dpi=600, bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig(save_as, dpi=sizes[0])
        else:
            plt.show()
        plt.close()
    elif ms.is_3D:
        if save_as is None:
            raise ValueError('3D microstructures must be saved in order to view them.')
        vtk_filename = f'{save_as[:-4]}.vtr'
        if os.path.isfile(vtk_filename):
            os.remove(vtk_filename)
        ms.to_paraview(vtk_filename)
        try:
            subprocess.run(['paraview', vtk_filename]) 
        except Exception:
            logging.info(f'Tried to open {vtk_filename} in paraview, but an error occurred. Try to open it yourself.')
    else:
        raise ValueError('The shape of the MS should be 2 or 3')


def main(args):
    """Main function for viewing script. This wraps some i/o around the view() function in order to make it usable
    via a command line interface and via a GUI. When mcrpy is used as a Python module directly, this main function is not called."""
    if not os.path.isfile(args.infile):
        raise ValueError(f'Given file {args.infile} does not exist!')

    target_folder, _ = os.path.split(args.infile)
    target_folder = '.' if target_folder == '' else target_folder
    args.logfile_additives = ''
    log.setup_logging(target_folder, args)
    save_as = args.infile if args.savefig else None
    if args.symmetry is not None and not isinstance(args.symmetry, Symmetry):
        assert args.symmetry in symmetries
        args.symmetry = symmetries[args.symmetry]

    if args.infile.endswith('.pickle'):
        with open(args.infile, 'rb') as f:
            data = pickle.load(f)
        if save_as and not isinstance(data, Microstructure):
            save_as = f'{save_as[:-7]}_D.png'
    elif args.infile.endswith('.npy'):
        save_as = f'{save_as[:-4]}_MS.png' if save_as else None
        data = Microstructure.from_npy(args.infile)
    else:
        raise NotImplementedError('Filetype not supported')

    if args.original_ms is not None:
        if not os.path.isfile(args.original_ms):
            raise ValueError('Provided original_ms {original_ms}, but file does not exist.')
        args.original_ms = Microstructure.from_npy(args.original_ms)

    view(data, save_as=save_as, original_ms=args.original_ms, axis=not args.noaxis, cmap=args.cmap, symmetry = args.symmetry)

def view(
        data: Union[np.ndarray, Dict[str, Any]],
        save_as: str = None,
        original_ms: Microstructure = None,
        axis: bool = True,
        cmap: str = 'cividis',
        symmetry: Symmetry = None):
    if isinstance(data, Microstructure):
        view_microstructure(data, save_as, axis=axis, cmap=cmap, symmetry=symmetry)
    elif isinstance(data, dict):
        view_generic_pickle(data, save_as, original_ms)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str)
    parser.add_argument('--original_ms', type=str, default=None)
    parser.add_argument('--cmap', type=str, default='cividis')
    parser.add_argument('--symmetry', type=str, help='Symmetry of the microstructure if orientations are considered. Default is None.', default=None)
    parser.add_argument('--logfile_name', type=str, default='logfile')
    parser.add_argument('--logging_level', type=int, default=logging.INFO)
    parser.add_argument('--logfile_date', dest='logfile_date', action='store_true')
    parser.set_defaults(logfile_date=False)
    parser.add_argument('--savefig', dest='savefig', action='store_true')
    parser.set_defaults(savefig=False)
    parser.add_argument('--noaxis', dest='noaxis', action='store_true')
    parser.set_defaults(noaxis=False)
    args = parser.parse_args()
    main(args)
