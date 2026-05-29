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

from abc import ABC, abstractmethod
import logging

import numpy as np
import tensorflow as tf

from mcrpy.descriptors.Descriptor import Descriptor
from mcrpy.src.SHSH import make_shsh_encoding
from mcrpy.orientation import Orientation, Rodrigues, Hypersphere, UnnormalizedQuaternion
from mcrpy.src.Symmetry import Symmetry, Cubic


class OrientationDescriptor(Descriptor):
    required_orientation_representation = None  # none means SHSH
    orientation_representation_dimension: int = 9

    @classmethod
    def make_descriptor(
        cls,
        desired_shape_2d=None,
        desired_shape_extended=None,
        use_multigrid_descriptor=True,
        symmetry: Symmetry = Cubic,
        n_shsh_terms: int = None,
        limit_to=8,
        **kwargs,
    ) -> callable:
        """By default wraps self.make_single_phase_descriptor."""

        assert symmetry is not None, "Crystallographic symmetry must be defined"

        if use_multigrid_descriptor:
            singlephase_descriptor = cls.make_multigrid_descriptor(
                limit_to=limit_to,
                desired_shape_2d=desired_shape_2d,
                desired_shape_extended=desired_shape_extended,
                symmetry=symmetry,
                n_shsh_terms=n_shsh_terms,
                **kwargs,
            )
        else:
            singlephase_descriptor = cls.make_singlegrid_descriptor(
                limit_to=limit_to,
                desired_shape_2d=desired_shape_2d,
                desired_shape_extended=desired_shape_extended,
                symmetry=symmetry,
                n_shsh_terms=n_shsh_terms,
                **kwargs,
            )
        ms_shape = desired_shape_extended
        if cls.required_orientation_representation is None:
            ori_converter = make_shsh_encoding(symmetry, n_shsh_terms, desired_shape_2d=desired_shape_2d)
        elif cls.required_orientation_representation == UnnormalizedQuaternion:
            ori_converter = lambda x: x
        else:
            ori_converter = lambda x: symmetry.project_to_fz(x.astype(cls.required_orientation_representation))

        @tf.function
        def singlephase_wrapper(x: tf.Tensor) -> tf.Tensor:
            ori_conv = ori_converter(x)
            phase_descriptor = tf.expand_dims(singlephase_descriptor(ori_conv), axis=0)
            return phase_descriptor

        return singlephase_wrapper

    @classmethod
    def make_singlephase_descriptor(cls, **kwargs) -> callable:
        """Rename to make_orientation_descriptor for easier naming."""
        return cls.make_orientation_descriptor(**kwargs)

    @classmethod
    def make_orientation_descriptor(cls, **kwargs) -> callable:
        """Analogous to make_singlephase_descriptor for phase microstructures."""
        raise NotImplementedError("Implement this in all used Descriptor subclasses")
