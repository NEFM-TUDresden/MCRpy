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

import numpy as np
import tensorflow as tf

from mcrpy.src import descriptor_factory
from mcrpy.descriptors.MultiPhaseDescriptor import MultiPhaseDescriptor
from mcrpy.descriptors.SPB import compute_phase_boundaries
from mcrpy.descriptors.Descriptor import make_image_padder


class TPB(MultiPhaseDescriptor):
    """Triple-Phase boundary density in 2D, normalized so it is a proper descriptor """

    is_differentiable = True
    default_weight = 100.0

    @staticmethod
    def make_multiphase_descriptor(n_phases: int = None, **kwargs) -> callable:
        """Make a periodic variation descriptor. This is not the total variation but the
        mean variation, since the total variation would not be a proper descriptor to compare
        across different resolutions."""

        tile_img = make_image_padder(1, 1)

        @tf.function
        def periodic_variation(img: tf.Tensor) -> tf.Tensor:
            imgs_tiled = [tile_img(tf.gather(img, [n_phase], axis=-1)) for n_phase in range(n_phases)]
            area = tf.cast(tf.size(img) / n_phases, tf.float64)
            phase_boundaries = [compute_phase_boundaries(img_tiled) for img_tiled in imgs_tiled]
            results_list = []
            for phase_1 in range(n_phases):
                pb_1 = phase_boundaries[phase_1]
                for phase_2 in range(phase_1 + 1, n_phases):
                    pb_2 = phase_boundaries[phase_2]
                    for phase_3 in range(phase_2 + 1, n_phases):
                        pb_3 = phase_boundaries[phase_3]
                        pb_joint = pb_1 * pb_2 * pb_3
                        int_phase_boundary = tf.reduce_sum(pb_joint)
                        boundary_fraction = int_phase_boundary / area
                        results_list.append(boundary_fraction)
            results = tf.stack(results_list)
            return results

        return periodic_variation


def register() -> None:
    descriptor_factory.register("TPB", TPB)
