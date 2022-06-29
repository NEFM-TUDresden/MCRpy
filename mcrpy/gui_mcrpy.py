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
import gooey
import logging

from mcrpy.characterize import main as main_characterize
from mcrpy.reconstruct import main as main_reconstruct
from mcrpy.match import main as main_match
from mcrpy.merge import main as main_merge
from mcrpy.interpolate import main as main_interpolate
from mcrpy.view import main as main_view
from mcrpy.smooth import main as main_smooth
from mcrpy.src.descriptor_factory import descriptor_choices
from mcrpy.src.loss_factory import loss_choices
from mcrpy.src.optimizer_factory import optimizer_choices

DARK = False
COLOR_FONT = '#ffffff' if DARK else '#000000'
COLOR_BG = '#282828' if DARK else '#ffffff'
COLOR_BODY = '#333333' if DARK else '#dddddd'
COLOR_NEUTRAL = '#BBBBBB'

# DARK = False
# COLOR_FONT = '#ffffff' if DARK else '#000000'
# COLOR_BG = '#282828' if DARK else '#ffffff'
# COLOR_BODY = '#333333' if DARK else '#fffff'
# COLOR_NEUTRAL = '#fffff'

GOOEY_OPTIONS = {
    'label_color': COLOR_FONT,
    'help_color': COLOR_FONT,
        }

def str2bool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif string.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{string} is not a valid boolean value')


@gooey.Gooey(
        advanced = True,
        monospace_display = True,
        header_bg_color = COLOR_NEUTRAL,
        header_font_color = COLOR_FONT,
        footer_bg_color = COLOR_BODY,
        footer_font_color = COLOR_FONT,
        body_bg_color = COLOR_BODY,
        sidebar_bg_color = COLOR_NEUTRAL,
        terminal_panel_color = '#282828',
        terminal_font_color = '#ffffff',
        default_size = (900, 700),
        program_name = "MCRpy",
        show_sidebar=True,
        clear_before_run=True,
        progress_regex=r"^Iteration (?P<current>\d+) of (?P<total>\d+)",
        progress_expr="current / total * 100",
        )
def call_main():
    parser = gooey.GooeyParser(description='Microstructure Characterization and Reconstruction in Python')
    subparsers = parser.add_subparsers()

    parser_cha = subparsers.add_parser('Characterize')
    parser_cha.set_defaults(func=main_characterize)
    group_cha_req = parser_cha.add_argument_group("Required options", "", gooey_options=GOOEY_OPTIONS)
    group_cha_req.add_argument('microstructure_filename', type=str, help='File of microstructure to reconstruct', widget='FileChooser', gooey_options=GOOEY_OPTIONS)
    group_cha_req.add_argument('--isotropic', type=str2bool, dest='isotropic', choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_cha_req.set_defaults(isotropic=False)
    group_cha_des = parser_cha.add_argument_group("Descriptor options", "", gooey_options=GOOEY_OPTIONS)
    group_cha_des.add_argument('--descriptor_types', nargs='+', type=str, help='Descriptor types (list)', default=['Correlations'], choices=descriptor_choices, widget='Listbox', gooey_options=GOOEY_OPTIONS)
    group_cha_des.add_argument('--slice_mode', type=str, help='Average or sample slices?', default='average', widget='Dropdown', choices=['average', 'sample', 'sample_surface'], gooey_options=GOOEY_OPTIONS)
    group_cha_des.add_argument('--nl_method', type=str, help='Nonlinearity method.', default='relu', choices=['relu', 'silu', 'gelu', 'elu', 'leaky_relu'], widget='Dropdown', gooey_options=GOOEY_OPTIONS)
    group_cha_des.add_argument('--limit_to', type=int, help='Limit in pixels to which to limit the characterisation metrics.', default=16, gooey_options=GOOEY_OPTIONS)
    group_cha_des.add_argument('--threshold_steepness', type=float, help='Steepness of soft threshold function. Regularisation parameter.', default=10.0, gooey_options=GOOEY_OPTIONS)
    group_cha_des.add_argument('--gram_weights_filename', type=str, help='Gram weigths filename wo path.', default='vgg19_normalized.pkl')
    group_cha_des.add_argument('--use_multiphase', type=str2bool, dest='use_multiphase', choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_cha_des.set_defaults(use_multiphase=False)
    group_cha_des.add_argument('--grey_values', dest='grey_values', action='store_true')
    group_cha_des.set_defaults(grey_values=False)
    group_cha_des.add_argument('--use_multigrid_descriptor', type=str2bool, dest='use_multigrid_descriptor', choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_cha_des.set_defaults(use_multigrid_descriptor=True)
    group_cha_com = parser_cha.add_argument_group("Common options", "", gooey_options=GOOEY_OPTIONS)
    group_cha_com.add_argument('--data_folder', type=str, help='Results folder. If None, default with timestamp.', default='results', widget='DirChooser', gooey_options=GOOEY_OPTIONS)
    group_cha_com.add_argument('--information', type=str, help='Information that is added to files that are written.', default=None, gooey_options=GOOEY_OPTIONS)
    group_cha_com.add_argument('--precision', type=int, help='Precision. 64 recommended', default=64, widget='Dropdown', choices=[32, 64], gooey_options=GOOEY_OPTIONS)
    group_cha_com.add_argument('--logfile_name', type=str, help='Name of logfile w/o extension.', default='logfile', gooey_options=GOOEY_OPTIONS)
    group_cha_com.add_argument('--logging_level', type=int, help='Logging level.', default=logging.INFO, gooey_options=GOOEY_OPTIONS)
    group_cha_com.add_argument('--logfile_date', type=str2bool, dest='logfile_date', choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_cha_com.set_defaults(logfile_date=False)

    parser_rec = subparsers.add_parser('Reconstruct')
    parser_rec.set_defaults(func=main_reconstruct)
    group_rec_req = parser_rec.add_argument_group("Required options", "", gooey_options=GOOEY_OPTIONS)
    group_rec_req.add_argument('descriptor_filename', type=str, help='File of descriptor to reconstruct from', widget='FileChooser', gooey_options=GOOEY_OPTIONS)
    group_rec_req.add_argument('extent_x', type=int, help='Extent in x-direction in pixels.', default=64, gooey_options=GOOEY_OPTIONS)
    group_rec_req.add_argument('extent_y', type=int, help='Extent in y-direction in pixels.', default=64, gooey_options=GOOEY_OPTIONS)
    group_rec_req.add_argument('extent_z', type=int, help='Extent in z-direction in pixels.', default=1, gooey_options=GOOEY_OPTIONS)
    group_rec_des = parser_rec.add_argument_group("Descriptor options", "", gooey_options=GOOEY_OPTIONS)
    group_rec_des.add_argument('--descriptor_types', nargs='+', type=str, help='Descriptor types (list)', default=['Correlations'], choices=descriptor_choices, widget='Listbox', gooey_options=GOOEY_OPTIONS)
    group_rec_des.add_argument('--nl_method', type=str, help='Nonlinearity method.', default='relu', choices=['relu', 'silu', 'gelu', 'elu', 'leaky_relu'], widget='Dropdown', gooey_options=GOOEY_OPTIONS)
    group_rec_des.add_argument('--limit_to', type=int, help='Limit in pixels to which to limit the characterisation metrics.', default=16, gooey_options=GOOEY_OPTIONS)
    group_rec_des.add_argument('--threshold_steepness', type=float, help='Steepness of soft threshold function. Regularisation parameter.', default=10.0, gooey_options=GOOEY_OPTIONS)
    group_rec_des.add_argument('--gram_weights_filename', type=str, help='Gram weigths filename wo path.', default='vgg19_normalized.pkl')
    group_rec_des.add_argument('--grey_values', dest='grey_values', action='store_true')
    group_rec_des.set_defaults(grey_values=False)
    group_rec_des.add_argument('--use_multiphase', type=str2bool, dest='use_multiphase', choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_rec_des.set_defaults(use_multiphase=False)
    group_rec_des.add_argument('--use_multigrid_descriptor', type=str2bool, dest='use_multigrid_descriptor', choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_rec_des.set_defaults(use_multigrid_descriptor=True)
    group_rec_los = parser_rec.add_argument_group("Loss options", "", gooey_options=GOOEY_OPTIONS)
    group_rec_los.add_argument('--loss_type', type=str, help='Loss type defined in loss plugins.', default='MSE', choices=loss_choices, widget='Dropdown', gooey_options=GOOEY_OPTIONS)
    group_rec_los.add_argument('--descriptor_weights', nargs='+', type=float, help='Descriptor weights (list)', default=None, gooey_options=GOOEY_OPTIONS)
    group_rec_los.add_argument('--oor_multiplier', type=float, help='Penalty weight for OOR in loss', default=1000.0, gooey_options=GOOEY_OPTIONS)
    group_rec_los.add_argument('--phase_sum_multiplier', type=float, help='Penalty weight for phase sum in loss', default=1000.0, gooey_options=GOOEY_OPTIONS)
    group_rec_opt = parser_rec.add_argument_group("Optimizer options", "", gooey_options=GOOEY_OPTIONS)
    group_rec_opt.add_argument('--optimizer_type', type=str, help='Optimizer type defined in optimizer plugins.', default='LBFGSB', choices=optimizer_choices, widget='Dropdown', gooey_options=GOOEY_OPTIONS)
    group_rec_opt.add_argument('--learning_rate', type=float, help='Learning rate of optimizer.', default=0.01, gooey_options=GOOEY_OPTIONS)
    group_rec_opt.add_argument('--beta_1', type=float, help='beta_1 parameter of optimizer.', default=0.9, gooey_options=GOOEY_OPTIONS)
    group_rec_opt.add_argument('--beta_2', type=float, help='beta_2 parameter of optimizer.', default=0.999, gooey_options=GOOEY_OPTIONS)
    group_rec_opt.add_argument('--rho', type=float, help='rho parameter of optimizer.', default=0.9, gooey_options=GOOEY_OPTIONS)
    group_rec_opt.add_argument('--momentum', type=float, help='momentum parameter of optimizer.', default=0.0, gooey_options=GOOEY_OPTIONS)
    group_rec_opt.add_argument('--initial_temperature', type=float, help='Initial temperature for annealing', default=0.004, gooey_options=GOOEY_OPTIONS)
    group_rec_opt.add_argument('--final_temperature', type=float, help='Final temperature for annealing', default=None, gooey_options=GOOEY_OPTIONS)
    group_rec_opt.add_argument('--cooldown_factor', type=float, help='Cooldown factor for annealing', default=0.9, gooey_options=GOOEY_OPTIONS)
    group_rec_rec = parser_rec.add_argument_group("Reconstruction options", "", gooey_options=GOOEY_OPTIONS)
    group_rec_rec.add_argument('--max_iter', type=int, help='Maximum number of iterations.', default=500, gooey_options=GOOEY_OPTIONS)
    group_rec_rec.add_argument('--convergence_data_steps', type=int, help='Each x steps write data', default=10, gooey_options=GOOEY_OPTIONS)
    group_rec_rec.add_argument('--outfile_data_steps', type=int, help='Each x steps write data to disk', default=None, gooey_options=GOOEY_OPTIONS)
    group_rec_rec.add_argument('--tolerance', type=float, help='Resonctruction tolerance.', default=1e-10, gooey_options=GOOEY_OPTIONS)
    group_rec_rec.add_argument('--stride', type=int, help='Stride (3D DiffMCR).', default=4, widget='Dropdown', choices=[1, 2, 4, 8, 16, 32, 64], gooey_options=GOOEY_OPTIONS)
    group_rec_rec.add_argument('--use_multigrid_reconstruction', type=str2bool, dest='use_multigrid_reconstruction', choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_rec_rec.set_defaults(use_multigrid_reconstruction=False)
    group_rec_com = parser_rec.add_argument_group("Common options", "", gooey_options=GOOEY_OPTIONS)
    group_rec_com.add_argument('--data_folder', type=str, help='Results folder. If None, default with timestamp.', default='results', widget='DirChooser', gooey_options=GOOEY_OPTIONS)
    group_rec_com.add_argument('--initial_microstructure_file', type=str, help='Microstructure to initializa with. If None, random initialization is used', default=None, widget='FileChooser', gooey_options=GOOEY_OPTIONS)
    group_rec_com.add_argument('--information', type=str, help='Information that is added to files that are written.', default=None, gooey_options=GOOEY_OPTIONS)
    group_rec_com.add_argument('--precision', type=int, help='Precision. 64 recommended', default=64, widget='Dropdown', choices=[32, 64], gooey_options=GOOEY_OPTIONS)
    group_rec_com.add_argument('--logfile_name', type=str, help='Name of logfile w/o extension.', default='logfile', gooey_options=GOOEY_OPTIONS)
    group_rec_com.add_argument('--logging_level', type=int, help='Logging level.', default=logging.INFO, gooey_options=GOOEY_OPTIONS)
    group_rec_com.add_argument('--logfile_date', dest='logfile_date', type=str2bool, choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_rec_com.set_defaults(logfile_date=False)

    parser_mat = subparsers.add_parser('Match')
    parser_mat.set_defaults(func=main_match)
    group_mat_req = parser_mat.add_argument_group("Required options", "", gooey_options=GOOEY_OPTIONS)
    group_mat_req.add_argument('microstructure_filename', type=str, help='File of microstructure to reconstruct', widget='FileChooser', gooey_options=GOOEY_OPTIONS)
    group_mat_req.add_argument('--isotropic', dest='isotropic', help='If the structure is 3D, should it be considered as isotropic?', type=str2bool, choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_mat_req.set_defaults(isotropic=False)
    group_mat_req.add_argument('--add_dimension', type=int, help='If the original structure is 2D, it can be reconstructed in 3D by adding a third dimension. Provide number of pixels in third dimension or nothing', default=None, gooey_options=GOOEY_OPTIONS)
    group_mat_des = parser_mat.add_argument_group("Descriptor options", "", gooey_options=GOOEY_OPTIONS)
    group_mat_des.add_argument('--descriptor_types', nargs='+', type=str, help='List of descriptor types. Shift-click to select multiple different options.', default=['Correlations'], choices=descriptor_choices, widget='Listbox', gooey_options=GOOEY_OPTIONS)
    group_mat_des.add_argument('--limit_to', type=int, help='Limit in pixels for highest-accuracy descriptors. Higher number is more accurate but less efficient.', default=16, gooey_options=GOOEY_OPTIONS)
    group_mat_des.add_argument('--use_multiphase', type=str2bool, help='Should multiple phase indicator functions be considered? Required for more than 2 phases but also possible for 2 phases.', dest='use_multiphase', choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_mat_des.set_defaults(use_multiphase=False)
    group_mat_des.add_argument('--use_multigrid_descriptor', help='Should a multigrid scheme be used for computing the descriptors? This is required for but does not automatically enable multigrid reconstruction.', type=str2bool, dest='use_multigrid_descriptor', choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_mat_des.set_defaults(use_multigrid_descriptor=True)
    group_mat_des.add_argument('--nl_method', type=str, help='Nonlinearity method.', default='relu', choices=['relu', 'silu', 'gelu', 'elu', 'leaky_relu'], widget='Dropdown', gooey_options=GOOEY_OPTIONS)
    group_mat_des.add_argument('--threshold_steepness', type=float, help='Steepness of soft threshold function. Regularisation parameter.', default=10.0, gooey_options=GOOEY_OPTIONS)
    group_mat_des.add_argument('--gram_weights_filename', type=str, help='Gram weigths filename wo path.', default='vgg19_normalized.pkl')
    group_mat_des.add_argument('--grey_values', dest='grey_values', action='store_true')
    group_mat_des.set_defaults(grey_values=False)
    group_mat_los = parser_mat.add_argument_group("Loss options", "", gooey_options=GOOEY_OPTIONS)
    group_mat_los.add_argument('--loss_type', type=str, help='Loss type defined in loss plugins.', default='MSE', choices=loss_choices, widget='Dropdown', gooey_options=GOOEY_OPTIONS)
    group_mat_los.add_argument('--descriptor_weights', nargs='+', type=float, help='Descriptor weights (list)', default=None, gooey_options=GOOEY_OPTIONS)
    group_mat_los.add_argument('--oor_multiplier', type=float, help='Penalty weight for OOR in loss', default=1000.0, gooey_options=GOOEY_OPTIONS)
    group_mat_los.add_argument('--phase_sum_multiplier', type=float, help='Penalty weight for phase sum in loss', default=1000.0, gooey_options=GOOEY_OPTIONS)
    group_mat_opt = parser_mat.add_argument_group("Optimizer options", "", gooey_options=GOOEY_OPTIONS)
    group_mat_opt.add_argument('--optimizer_type', type=str, help='Optimizer type defined in optimizer plugins.', default='LBFGSB', choices=optimizer_choices, widget='Dropdown', gooey_options=GOOEY_OPTIONS)
    group_mat_opt.add_argument('--learning_rate', type=float, help='Learning rate of optimizer.', default=0.01, gooey_options=GOOEY_OPTIONS)
    group_mat_opt.add_argument('--beta_1', type=float, help='beta_1 parameter of optimizer.', default=0.9, gooey_options=GOOEY_OPTIONS)
    group_mat_opt.add_argument('--beta_2', type=float, help='beta_2 parameter of optimizer.', default=0.999, gooey_options=GOOEY_OPTIONS)
    group_mat_opt.add_argument('--rho', type=float, help='rho parameter of optimizer.', default=0.9, gooey_options=GOOEY_OPTIONS)
    group_mat_opt.add_argument('--momentum', type=float, help='momentum parameter of optimizer.', default=0.0, gooey_options=GOOEY_OPTIONS)
    group_mat_opt.add_argument('--initial_temperature', type=float, help='Initial temperature for annealing', default=0.004, gooey_options=GOOEY_OPTIONS)
    group_mat_opt.add_argument('--final_temperature', type=float, help='Final temperature for annealing', default=None, gooey_options=GOOEY_OPTIONS)
    group_mat_opt.add_argument('--cooldown_factor', type=float, help='Cooldown factor for annealing', default=0.9, gooey_options=GOOEY_OPTIONS)
    group_mat_rec = parser_mat.add_argument_group("Reconstruction options", "", gooey_options=GOOEY_OPTIONS)
    group_mat_rec.add_argument('--max_iter', type=int, help='Maximum number of iterations.', default=500, gooey_options=GOOEY_OPTIONS)
    group_mat_rec.add_argument('--convergence_data_steps', type=int, help='Each x steps write data', default=10, gooey_options=GOOEY_OPTIONS)
    group_mat_rec.add_argument('--outfile_data_steps', type=int, help='Each x steps write data to disk', default=None, gooey_options=GOOEY_OPTIONS)
    group_mat_rec.add_argument('--tolerance', type=float, help='Resonctruction tolerance.', default=1e-10, gooey_options=GOOEY_OPTIONS)
    group_mat_rec.add_argument('--stride', type=int, help='Stride (3D DiffMCR).', default=4, widget='Dropdown', choices=[1, 2, 4, 8, 16, 32, 64], gooey_options=GOOEY_OPTIONS)
    group_mat_rec.add_argument('--use_multigrid_reconstruction', dest='use_multigrid_reconstruction', type=str2bool, choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_mat_rec.set_defaults(use_multigrid_reconstruction=False)
    group_mat_com = parser_mat.add_argument_group("Common options", "", gooey_options=GOOEY_OPTIONS)
    group_mat_com.add_argument('--data_folder', type=str, help='Results folder. If None, default with timestamp.', default='results', widget='DirChooser', gooey_options=GOOEY_OPTIONS)
    group_mat_com.add_argument('--initial_microstructure_file', type=str, help='Microstructure to initializa with. If None, random initialization is used', default=None, widget='FileChooser', gooey_options=GOOEY_OPTIONS)
    group_mat_com.add_argument('--information', type=str, help='Information that is added to files that are written.', default=None, gooey_options=GOOEY_OPTIONS)
    group_mat_com.add_argument('--precision', type=int, help='Precision. 64 recommended', default=64, widget='Dropdown', choices=[32, 64], gooey_options=GOOEY_OPTIONS)
    group_mat_com.add_argument('--logfile_name', type=str, help='Name of logfile w/o extension.', default='logfile', gooey_options=GOOEY_OPTIONS)
    group_mat_com.add_argument('--logging_level', type=int, help='Logging level.', default=logging.INFO, gooey_options=GOOEY_OPTIONS)
    group_mat_com.add_argument('--logfile_date', dest='logfile_date', type=str2bool, choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_mat_com.set_defaults(logfile_date=False)

    parser_vie = subparsers.add_parser('View')
    parser_vie.set_defaults(func=main_view)
    group_vie_req = parser_vie.add_argument_group("Required options", "", gooey_options=GOOEY_OPTIONS)
    group_vie_req.add_argument('infile', type=str, help='File to view. Can be (1) a .npy-file containing a microstructure, (2) a .pickle-file containing a characterization or (3) a .pickle-file containing convergence data. ', widget='FileChooser', gooey_options=GOOEY_OPTIONS)
    group_vie_opt = parser_vie.add_argument_group("Additional options", "", gooey_options=GOOEY_OPTIONS)
    group_vie_opt.add_argument('--original_ms', type=str, help='If convergence data is plotted, this adds the original microstructure for easy comparison.', widget='FileChooser', gooey_options=GOOEY_OPTIONS)
    group_vie_opt.add_argument('--logfile_name', type=str, help='Name of logfile w/o extension.', default='logfile', gooey_options=GOOEY_OPTIONS)
    group_vie_opt.add_argument('--logging_level', type=int, help='Logging level.', default=logging.INFO, gooey_options=GOOEY_OPTIONS)
    group_vie_opt.add_argument('--logfile_date', dest='logfile_date', type=str2bool, choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_vie_opt.set_defaults(logfile_date=False)
    group_vie_opt.add_argument('--savefig', dest='savefig', type=str2bool, choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_vie_opt.set_defaults(savefig=False)

    parser_smo = subparsers.add_parser('Smooth')
    parser_smo.set_defaults(func=main_smooth)
    group_smo_req = parser_smo.add_argument_group("Required options", "", gooey_options=GOOEY_OPTIONS)
    group_smo_req.add_argument('microstructure_filename', type=str, help='File of microstructure to reconstruct', widget='FileChooser', gooey_options=GOOEY_OPTIONS)
    group_smo_opt = parser_smo.add_argument_group("Additional options", "", gooey_options=GOOEY_OPTIONS)
    group_smo_opt.add_argument('--method', type=str, help='Smoothing algorithm', default='gaussian', choices=['gaussian'], widget='Dropdown', gooey_options=GOOEY_OPTIONS)
    group_smo_opt.add_argument('--strength', type=float, help='Smoothing strength', default=1.0, gooey_options=GOOEY_OPTIONS)
    group_smo_opt.add_argument('--info', type=str, help='Information that is added to files that are written.', default=None, gooey_options=GOOEY_OPTIONS)

    parser_mer = subparsers.add_parser('Merge')
    parser_mer.set_defaults(func=main_merge)
    group_mer_req = parser_mer.add_argument_group("Required options", "", gooey_options=GOOEY_OPTIONS)
    group_mer_req.add_argument('x_file', type=str, help='File containing descriptor in x-plane', widget='FileChooser', gooey_options=GOOEY_OPTIONS)
    group_mer_req.add_argument('y_file', type=str, help='File containing descriptor in y-plane', widget='FileChooser', gooey_options=GOOEY_OPTIONS)
    group_mer_req.add_argument('--z_file', type=str, help='File containing descriptor in z-plane', widget='FileChooser', gooey_options=GOOEY_OPTIONS, default=None)
    group_mer_req.add_argument('outfile', type=str, help='File to write merged descriptor to', gooey_options=GOOEY_OPTIONS)

    parser_int = subparsers.add_parser('Interpolate')
    parser_int.set_defaults(func=main_interpolate)
    group_int_req = parser_int.add_argument_group("Required options", "", gooey_options=GOOEY_OPTIONS)
    group_int_req.add_argument('microstructure_descriptor_1', type=str, help='File of first microstructure descriptor to interpolate from', widget='FileChooser', gooey_options=GOOEY_OPTIONS)
    group_int_req.add_argument('microstructure_descriptor_2', type=str, help='File of second microstructure descriptor to interpolate to', widget='FileChooser', gooey_options=GOOEY_OPTIONS)
    group_int_req.add_argument('--n_steps', type=int, help='Number of steps for interpolation. This includes start and end point, so the minimal value is 3.', default=3, gooey_options=GOOEY_OPTIONS)
    group_int_opt = parser_int.add_argument_group("Additional options", "", gooey_options=GOOEY_OPTIONS)
    group_int_opt.add_argument('--data_folder', type=str, help='Results folder. If None, default with timestamp.', default='results', widget='DirChooser', gooey_options=GOOEY_OPTIONS)
    group_int_opt.add_argument('--logfile_name', type=str, help='Name of logfile w/o extension.', default='logfile', gooey_options=GOOEY_OPTIONS)
    group_int_opt.add_argument('--logging_level', type=int, help='Logging level.', default=logging.INFO, gooey_options=GOOEY_OPTIONS)
    group_int_opt.add_argument('--logfile_date', dest='logfile_date', type=str2bool, choices=[True, False], gooey_options=GOOEY_OPTIONS)
    group_int_opt.set_defaults(logfile_date=False)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    call_main()

