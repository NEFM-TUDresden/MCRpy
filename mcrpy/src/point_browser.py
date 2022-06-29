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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np

from typing import Tuple

from mcrpy.src import fileutils

mplstyle.use('seaborn-notebook')

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
            raw_data: np.ndarray,
            line_datas: np.ndarray = None,
            original_ms: np.ndarray = None,
            xlabel: str = 'Iteration',
            ylabel: str = 'Cost',
            log_axis: bool = True):
        """Point browser to analyze MCR convergence data interactively.

        Args:
            scatter_data (np.ndarray): Points that will be scatter-plotted and should be clickable
                to obtain the MS at that iteration. First index is sample number, second index is
                0: n_iteration, 1: value.
            detail_data (np.ndarray): MS at different stages of convergence and additional information
                (like correlation fields or so). First index is sample number, second is field number
                (currently only size 1), third, fourth and fifth is 3D field. If field is 2D, fifth
                index will be added automatically.
            line_datas (dict, optional): Dict containing lines to plot. Keys are labels and values are
                np.ndarray with same format as scatter_data, but not necessarily the same number of
                samples. Defaults to None.
            layout (tuple, optional): Layout of the different plots. Defaults to (1, 2).
            log_axis (bool, optional): Use logarithmic axes for convergence plot. Recommended.
                Defaults to True.
        """
        self.raw_data = raw_data[:, 0]
        self.n_phases = self.raw_data.shape[-1]
        if len(self.raw_data.shape) == 5:
            self.raw_data = self.raw_data.reshape([*(self.raw_data.shape[:-1])] + [1, self.n_phases])
        self.decoded_data = np.array([[fileutils.decode_ms(ms[0])] for ms in self.raw_data])
        self.show_cycle = self.n_phases 
        self.show_cycle_mod = self.n_phases + 1
        if len(self.decoded_data.shape) == 4:
            self.decoded_data = self.decoded_data.reshape([*self.decoded_data.shape] + [1])
        assert len(self.decoded_data.shape) == 5
        assert self.raw_data.shape[:-1] == self.decoded_data.shape
        self.slices = self.decoded_data.shape[-1]
        self.ind = self.slices // 2

        mg_level_starts, mg_level_indices = find_mg_layers(scatter_data[:, 0])
        annotation_indices = []
        for unique_index in np.unique(np.array(mg_level_indices)):
            annotation_index = mg_level_indices.index(unique_index)
            if annotation_index > 0:
                annotation_indices.append(annotation_index - 1)

        self.line_datas = line_datas
        self.xs = shift_by_mg_layer(scatter_data[:, 0])
        self.ys = scatter_data[:, 1]

        figsize = (8, 4) if original_ms is None else (12, 4)
        self.fig = plt.figure(figsize=figsize, tight_layout=True)
        layout = (1, 2) if original_ms is None else (1, 3)
        gs = gridspec.GridSpec(*layout)
        layout_start = 0 if original_ms is None else 1

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
        self.text_scroll = self.ax.text(0.95, 0.90, 'use mouse wheel to scroll',
                transform=self.ax.transAxes, va='top', ha='right')
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
            original_ms = (original_ms - np.min(original_ms)) / (np.max(original_ms) - np.min(original_ms))
            self.og_axis = self.fig.add_subplot(gs[0, 0])
            self.og_img = self.og_axis.imshow(original_ms, cmap='cividis')
            self.og_axis.get_xaxis().set_visible(False)
            self.og_axis.get_yaxis().set_visible(False)
        self.other_axes = [self.fig.add_subplot(gs[0, layout_start + 1])]
        self.ims = [other_axis.imshow(self.decoded_data[self.lastind, 
            n_axis, :, :, self.ind], cmap='cividis') 
            for n_axis, other_axis in enumerate(self.other_axes)]
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

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update_scroll()

    def onpress(self, event):
        inc = 0
        if self.lastind is None:
            return
        if event.key not in ('n', 'p', 'c', 'r', 'e', 'b'):
            return
        if event.key == 'b':
            self.lastind = 0
        if event.key == 'e':
            self.lastind = len(self.xs) - 1
        if event.key == 'n':
            self.lastind += 1
        if event.key == 'p':
            self.lastind -= 1
        if event.key == 'r':
            self.show_cycle = self.n_phases
        if event.key == 'c':
            self.show_cycle = (self.show_cycle + 1) % self.show_cycle_mod

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
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update_choice()

    def update_choice(self):
        if self.lastind is None:
            return
        dataind = self.lastind

        data_at_dataind = self.decoded_data[dataind] if self.show_cycle == self.n_phases else self.raw_data[dataind, ..., self.show_cycle]
        for n_axis, _ in enumerate(self.other_axes):
            self.detail_data_3d[n_axis] = data_at_dataind[n_axis]

        self.selected.set_visible(True)
        self.selected.set_data(self.xs[dataind], self.ys[dataind])

        self.text_choice.set_text('selected: %d' % self.xs[dataind])
        if self.show_cycle == self.n_phases:
            if self.n_phases == 1:
                self.text_cycle.set_text('rounded ms')
            else:
                self.text_cycle.set_text('max phase')
        else:
            self.text_cycle.set_text(f'phase {self.show_cycle}')
        self.fig.canvas.draw()
        self.update_scroll()

    def update_scroll(self):
        for n_axis, im in enumerate(self.ims):
            im.set_data(self.detail_data_3d[n_axis][:, :, self.ind])
            im.axes.figure.canvas.draw()
        self.text_scroll.set_text('slice {} of {}'.format(self.ind + 1, self.slices))
