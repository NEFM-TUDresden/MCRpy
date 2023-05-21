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

from mcrpy.descriptors.PhaseDescriptor import PhaseDescriptor

class MultiPhaseDescriptor(PhaseDescriptor):

    @classmethod
    def make_descriptor(
            cls, 
            desired_shape_2d=None, 
            desired_shape_extended=None, 
            use_multigrid_descriptor=True, 
            use_multiphase=True, 
            limit_to = 8,
            **kwargs) -> callable:
        """By default wraps self.make_single_phase_descriptor."""
        if use_multigrid_descriptor:
            multiphase_descriptor =  cls.make_multigrid_descriptor(
                limit_to=limit_to,
                desired_shape_2d=desired_shape_2d,
                desired_shape_extended=desired_shape_extended,
                **kwargs)
        else:
            multiphase_descriptor = cls.make_singlegrid_descriptor(
                limit_to=limit_to, 
                desired_shape_2d=desired_shape_2d,
                desired_shape_extended=desired_shape_extended,
                **kwargs) 

        n_phases = desired_shape_extended[-1]
        assert n_phases > 1 and use_multiphase

        @tf.function
        def call_descriptor(x: tf.Tensor) -> tf.Tensor:
            multiphase_descriptor_result = tf.expand_dims(multiphase_descriptor(x), axis=0)
            return multiphase_descriptor_result
        return call_descriptor


    @classmethod
    def make_singlephase_descriptor(
            cls, 
            **kwargs) -> callable:
        """Rename to make_multiphase_descriptor for easier naming."""
        return cls.make_multiphase_descriptor(**kwargs)

    @classmethod
    def make_multiphase_descriptor(
            cls, 
            **kwargs) -> callable:
        """Analogous to make_singlephase_descriptor for phase microstructures."""
        raise NotImplementedError("Implement this in all used Descriptor subclasses")

