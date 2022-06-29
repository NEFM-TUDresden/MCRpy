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

from typing import Tuple

import pymks
import numpy as np
from sklearn.pipeline import Pipeline


from mcrpy.src import descriptor_factory
from mcrpy.descriptors.Descriptor import Descriptor


class FFTCorrelations(Descriptor):
    is_differentiable = False

    @staticmethod
    def make_singlephase_descriptor(
            only_black_auto: bool = True,
            desired_shape_2d: Tuple[int] = (64, 64),
            limit_to: int = 8,
            **kwargs) -> callable:
        phases = (0, 1)
        n_phases = 2
        min_phase = min(phases)
        max_phase = max(phases)
        correlations = [[1, 1]] if only_black_auto else [
            [min_phase, p] for p in phases]
        model = Pipeline(steps=[
            ('discretize', pymks.PrimitiveTransformer(
                n_state=n_phases, min_=min_phase, max_=max_phase)),
            ('correlations', pymks.TwoPointCorrelation(
                periodic_boundary=True,
                cutoff=limit_to - 1,
                correlations=correlations
            ))
        ])

        desired_pymks_shape = tuple([1] + list(desired_shape_2d))

        def compute_descriptor(x: np.ndarray) -> np.ndarray:
            x_data = x.reshape(desired_pymks_shape)
            x_stats = model.transform(x_data).persist()
            x_stats = x_stats.compute()
            return x_stats
        return compute_descriptor

    @staticmethod
    def define_comparison_mask(
            desired_descriptor_shape: Tuple[int] = None, 
            limit_to: int = None, 
            **kwargs):
        assert len(desired_descriptor_shape) == 4
        assert desired_descriptor_shape[0] == 1
        assert desired_descriptor_shape[-1] == 1
        assert desired_descriptor_shape[1] == desired_descriptor_shape[2]
        desired_limit_to = desired_descriptor_shape[1] // 2 + 1

        if limit_to == desired_limit_to:
            return None, False

        larger_limit_to = max(limit_to, desired_limit_to)
        smaller_limit_to = min(limit_to, desired_limit_to)
        limit_delta = larger_limit_to - smaller_limit_to
        larger_n_elements = larger_limit_to * 2 - 1
        mask = np.zeros((1, larger_n_elements, larger_n_elements, 1), dtype=np.bool8)
        mask[:, limit_delta:-limit_delta, limit_delta:-limit_delta, :] = True
        return mask, limit_to > desired_limit_to

    @classmethod
    def visualize_subplot(
            cls,
            descriptor_value: np.ndarray,
            ax,
            descriptor_type: str = None,
            mg_level: int = None,
            n_phase: int = None):
        s2 = descriptor_value[0, :, :, 0]
        height, width = s2.shape
        if height != width:
            raise NotImplementedError('Non-square FFTCorrelations not implemented')
        limit_to = height // 2 + 1
        ax.imshow(s2, cmap='cividis')
        ax.set_title(f'S2: l={mg_level}, p={n_phase}')
        ax.set_xlabel(r'$r_x$ in Px')
        ax.set_ylabel(r'$r_y$ in Px')
        xticks = [0, limit_to - 1, 2 * (limit_to - 1)]
        yticks = [0, limit_to - 1, 2 * (limit_to - 1)]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([-limit_to + 1, 0, limit_to - 1])
        ax.set_yticklabels(reversed([-limit_to + 1, 0, limit_to - 1]))



def register() -> None:
    descriptor_factory.register("FFTCorrelations", FFTCorrelations)
