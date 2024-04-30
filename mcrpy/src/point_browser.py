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

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import tensorflow as tf

from typing import Tuple, List

from mcrpy.src import fileutils
from mcrpy.src.Microstructure import Microstructure
from mcrpy.src.Settings import ReconstructionSettings


# mplstyle.use('seaborn-notebook')

def find_mg_layers(xs: np.ndarray):
    mg_level_starts = []
    mg_level_indices = []
    for i, x_i in enumerate(xs):
        x_im1 = xs[i-1]
        if x_i - x_im1 < 0 and i > 0:
            mg_level_starts.append(x_im1)
        mg_level_indices.append(len(mg_level_starts))
    return mg_level_starts, mg_level_indices

def shift_by_mg_layer(xs: np.ndarray):
    mg_level_starts, mg_level_indices = find_mg_layers(xs)
    for i, mg_level_index in enumerate(mg_level_indices):
        for mg_level_start in mg_level_starts[:mg_level_index]:
            xs[i] += mg_level_start
    return xs

class PointBrowser(object):
    fixed_linestyles = {'Cost': 'k-', 'L2(S2)': 'k--'}
    
    def __init__(self,
            scatter_data: np.ndarray,
            intermediate_microstructures: List[Microstructure],
            line_datas: np.ndarray = None,
            original_ms: Microstructure = None,
            xlabel: str = 'Iteration',
            ylabel: str = 'Cost',
            settings: ReconstructionSettings = None,
            log_axis: bool = True):
        self.intermediate_microstructures = [im if isinstance(im, Microstructure) else Microstructure(im[0, 0], use_multiphase=settings.use_multiphase, skip_encoding=True) for im in intermediate_microstructures] # legacy conversion
        self.last_microstructure = self.intermediate_microstructures[-1]
        if self.last_microstructure.has_orientations:
            for im in self.intermediate_microstructures:
                im.x = im.symmetry.project_to_fz(im.ori).x
        self.show_cycle = self.last_microstructure.n_phases if self.last_microstructure.has_phases else 3
        self.show_cycle_mod = self.show_cycle + 1 if self.last_microstructure.n_phases > 1 else self.show_cycle
        self.current_dim = 0
        self.slices = self.intermediate_microstructures[0].spatial_shape[self.current_dim]
        self.current_slice_number = self.slices // 2
        self.raw = False
        self.n_ori_view = 0

        mg_level_starts, mg_level_indices = find_mg_layers(scatter_data[:, 0])
        annotation_indices = []
        for unique_index in np.unique(np.array(mg_level_indices)):
            annotation_index = mg_level_indices.index(unique_index)
            if annotation_index > 0:
                annotation_indices.append(annotation_index - 1)

        self.line_datas = line_datas
        self.xs = shift_by_mg_layer(scatter_data[:, 0])
        self.ys = scatter_data[:, 1]
        self.original_ms = original_ms

        figsize = (8, 4) if original_ms is None else (12, 4)
        self.fig = plt.figure(figsize=figsize, tight_layout=True)
        layout = (1, 2) if original_ms is None else (1, 3)
        gs = gridspec.GridSpec(*layout)
        layout_start = 0 if original_ms is None else 1

        self.ori_views = [
            ('rho_1', lambda x: x[..., 0]),
            ('rho_2', lambda x: x[..., 1]),
            ('rho_3', lambda x: x[..., 2]),
                ]
        self.n_ori_views = len(self.ori_views)

        # main ax
        self.ax = self.fig.add_subplot(gs[0, layout_start])
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if log_axis:
            self.ax.set_yscale('log')
        self.line, = self.ax.plot(self.xs, self.ys, 'o', picker=True)  # 5 points tolerance
        self.line.set_pickradius(15)
        self.lastind = 0
        self.text_choice = self.ax.text(0.95, 0.95, 'selected: none',
                transform=self.ax.transAxes, va='top', ha='right')
        if self.last_microstructure.is_3D:
            self.text_scroll = self.ax.text(0.95, 0.90, 'use mouse wheel to scroll',
                    transform=self.ax.transAxes, va='top', ha='right')
        if self.last_microstructure.has_orientations:
            self.text_view = self.ax.text(0.95, 0.80, 'use v and w to cycle views',
                    transform=self.ax.transAxes, va='top', ha='right')
        if self.last_microstructure.has_phases and self.last_microstructure.n_phases > 1:
            self.text_cycle = self.ax.text(0.95, 0.85, 'use c to cycle between phases',
                    transform=self.ax.transAxes, va='top', ha='right')
        for mg_jump, annotation_index in enumerate(annotation_indices):
            self.ax.text(self.xs[annotation_index], 0.7 * self.ys[annotation_index], f'MG jump {mg_jump + 1}',
                    va='top', ha='center')
        self.selected, = self.ax.plot([self.xs[0]], [self.ys[0]], 'o', ms=12,
                alpha=0.4, color='yellow', visible=True)
        if line_datas is not None:
            for label in line_datas.keys():
                if label in self.fixed_linestyles.keys():
                    self.ax.plot(shift_by_mg_layer(line_datas[label][:, 0]), line_datas[label][:, 1],
                            self.fixed_linestyles[label], label=label)
                else:
                    self.ax.plot(shift_by_mg_layer(line_datas[label][:, 0]), line_datas[label][:, 1],
                            label=label)
            if len(line_datas.keys()) > 1:
                self.ax.legend(loc='upper center')

        # Handle other axes providing additional information
        if original_ms is not None:
            if original_ms.is_3D:
                raise NotImplementedError('Displaying 3D original_ms not implemented')
            self.og_axis = self.fig.add_subplot(gs[0, 0])
            if original_ms.has_phases:
                self.og_img = self.og_axis.imshow(original_ms.decode_phases(), cmap='cividis')
            else:
                self.og_img = self.og_axis.imshow(self.ori_views[self.n_ori_view][1](self.original_ms.get_orientation_field().numpy()))
            self.og_axis.get_xaxis().set_visible(False)
            self.og_axis.get_yaxis().set_visible(False)
        self.other_axes = [self.fig.add_subplot(gs[0, layout_start + 1])]
        self.ims = [
            other_axis.imshow(
                self.get_relevant_slice(
                    self.intermediate_microstructures[self.lastind]
                ),
                cmap='cividis',
            )
            for other_axis in self.other_axes
        ]
        self.detail_data_3d = [None for other_axis in self.other_axes]
        for other_axis in self.other_axes:
            other_axis.get_xaxis().set_visible(False)
            other_axis.get_yaxis().set_visible(False)

        # Connect events and launch
        self.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.fig.canvas.mpl_connect('key_press_event', self.onpress)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.update_choice()
        plt.show()

    def get_relevant_slice(self, ms: Microstructure):
        relevant_slice = ms.get_slice(self.current_dim, self.current_slice_number) if ms.is_3D else ms.x
        if self.last_microstructure.has_phases:
            return ms.decode_phase_array(relevant_slice,
                    specific_phase = None if self.show_cycle == self.last_microstructure.n_phases else self.show_cycle,
                    raw = self.raw)
        else:
            return self.ori_views[self.n_ori_view][1](relevant_slice.numpy()[0])

    def onscroll(self, event):
        if event.button == 'up':
            self.current_slice_number = (self.current_slice_number + 1) % self.slices
        else:
            self.current_slice_number = (self.current_slice_number - 1) % self.slices
        self.update_scroll()

    def onpress(self, event):
        inc = 0
        if self.lastind is None:
            return
        if event.key not in 'npcrebxyzvw':
            return
        if event.key == 'b':
            self.lastind = 0
        elif event.key == 'c':
            self.show_cycle = (self.show_cycle + 1) % self.show_cycle_mod

        elif event.key == 'e':
            self.lastind = len(self.xs) - 1
        elif event.key == 'n':
            self.lastind += 1
        elif event.key == 'p':
            self.lastind -= 1
        elif event.key == 'r':
            self.raw = not self.raw
        elif event.key == 'v':
            self.n_ori_view += 1
        elif event.key == 'w':
            self.n_ori_view -= 1
        elif event.key == 'x':
            self.current_dim = 0
        elif event.key == 'y':
            self.current_dim = 1
        elif event.key == 'z':
            self.current_dim = 2
        self.n_ori_view = self.n_ori_view % self.n_ori_views
        self.lastind = np.clip(self.lastind, 0, len(self.xs) - 1)
        self.update_choice()

    def onpick(self, event):
        if event.artist != self.line:
            return True

        N = len(event.ind)
        if not N:
            return True

        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - self.xs[event.ind], y - self.ys[event.ind])
        indmin = distances.argmin()
        self.lastind = event.ind[indmin]

        self.update_choice()

    def update_choice(self):
        if self.lastind is None:
            return
        self.slices = self.intermediate_microstructures[self.lastind].spatial_shape[self.current_dim]
        if self.current_slice_number >= self.slices:
            self.current_slice_number = self.slices // 2

        self.selected.set_visible(True)
        self.selected.set_data(self.xs[self.lastind], self.ys[self.lastind])
        # if self.last_microstructure.has_orientations:
        #     if 'z_' in self.ori_views[self.n_ori_view][0] or 'rho' in self.ori_views[self.n_ori_view][0]:
        #         self.selected.set(norm=matplotlib.colors.Normalize(-1, 1), cmap='seismic')

        if self.last_microstructure.has_orientations:
            self.text_view.set_text(self.ori_views[self.n_ori_view][0])
        if self.last_microstructure.has_phases and self.last_microstructure.n_phases > 1:
            if self.show_cycle == self.last_microstructure.n_phases:
                self.text_cycle.set_text('max phase')
            else:
                self.text_cycle.set_text(f'phase {self.show_cycle}')
        self.fig.canvas.draw()
        self.update_scroll()

    def update_scroll(self):
        for im in self.ims:
            im.set_data(self.get_relevant_slice(self.intermediate_microstructures[self.lastind]))
            if self.last_microstructure.has_orientations and (
                'z_' in self.ori_views[self.n_ori_view][0]
                or 'rho' in self.ori_views[self.n_ori_view][0]
            ):
                im.set(norm=matplotlib.colors.Normalize(-1, 1), cmap='seismic')
        if self.last_microstructure.has_orientations and self.original_ms is not None:
            self.og_img.set_data(self.ori_views[self.n_ori_view][1](self.original_ms.get_orientation_field().numpy()))
            if self.last_microstructure.has_orientations and (
                'z_' in self.ori_views[self.n_ori_view][0]
                or 'rho' in self.ori_views[self.n_ori_view][0]
            ):
                self.og_img.set(norm=matplotlib.colors.Normalize(-1, 1), cmap='seismic')
        self.text_choice.set_text('selected: %d' % self.xs[self.lastind])
        if self.intermediate_microstructures[self.lastind].is_3D:
            self.text_scroll.set_text(
                f'slice {self.current_slice_number + 1} of {self.slices}'
            )
        self.fig.canvas.draw()
