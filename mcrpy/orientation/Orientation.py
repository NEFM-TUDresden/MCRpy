from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

import mcrpy
from mcrpy.src.translation import translate_2D

tf.keras.backend.set_floatx("float64")

EPS = tf.constant(1e-15, dtype=tf.float64)
P = -1.0


class Orientation(tf.experimental.ExtensionType):
    x: tf.Tensor
    n_dims: int = -1

    def __call__(self):
        return self.x

    def __getitem__(self, item):
        return self.x[item]

    def rotate(self, rotation: Orientation) -> Orientation:
        return self.to_quaternion().rotate(rotation).astype(type(self))

    def rotate_vector(self, vector: tf.Tensor) -> tf.Tensor:
        return self.to_quaternion().rotate_vector(vector)

    @classmethod
    def from_quaternion(cls, q: mcrpy.Orientation.Quaternion) -> Orientation:
        raise NotImplementedError

    def to_quaternion(self) -> mcrpy.orientation.Quaternion:
        raise NotImplementedError

    def to_axis(self) -> mcrpy.orientation.Axis:
        return self.to_quaternion().to_axis()

    def validate(self) -> bool:
        raise NotImplementedError

    def __validate__(self):
        tf.assert_equal(self.validate(), True)

    def numpy(self) -> np.ndarray:
        return self.x.numpy()

    def astype(self, orientation_type: type) -> Orientation:
        if isinstance(self, orientation_type):
            return self
        return orientation_type.from_quaternion(self.to_quaternion())

    def compute_grain_boundaries(self) -> tf.Tensor:
        self.__validate__()

        gb_list = []
        for translation_x, translation_y in [(1, 0), (0, 1), (1, 1)]:
            q_translated = (translate_2D(self.x, translation_x, translation_y)[0]).to_quaternion()
            ori_delta = self.rotate(q_translated.conjugate).to_axis().angle
            gb_list.append(ori_delta / 3.14)
        return tf.concat(gb_list, axis=-1)
