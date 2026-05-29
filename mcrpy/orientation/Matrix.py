"""Based on Rowenhorst et al 2015"""

from __future__ import annotations
import tensorflow as tf

import mcrpy.orientation.Quaternion
from mcrpy.orientation.Orientation import Orientation, P

EPS = 1e-12


class Matrix(Orientation):
    x: tf.Tensor
    n_dims: int = 9

    @classmethod
    def from_quaternion(cls, q: mcrpy.orientation.Quaternion) -> Matrix:
        q_bar = tf.math.square(q.q_0) - (tf.math.square(q.q_1) + tf.math.square(q.q_2) + tf.math.square(q.q_3))
        a_11 = q_bar + 2.0 * tf.math.square(q.q_1)
        a_12 = 2.0 * (q.q_1 * q.q_2 - P * q.q_0 * q.q_3)
        a_13 = 2.0 * (q.q_1 * q.q_3 + P * q.q_0 * q.q_2)
        a_21 = 2.0 * (q.q_1 * q.q_2 + P * q.q_0 * q.q_3)
        a_22 = q_bar + 2.0 * tf.math.square(q.q_2)
        a_23 = 2.0 * (q.q_2 * q.q_3 - P * q.q_0 * q.q_1)
        a_31 = 2.0 * (q.q_1 * q.q_3 - P * q.q_0 * q.q_2)
        a_32 = 2.0 * (q.q_2 * q.q_3 + P * q.q_0 * q.q_1)
        a_33 = q_bar + 2.0 * tf.math.square(q.q_3)
        return cls(tf.concat([a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32, a_33], axis=-1))

    def to_quaternion(self) -> mcrpy.orientation.Quaternion:
        q_0 = 0.5 * tf.sqrt(1.0 + self.a_11 + self.a_22 + self.a_33)
        q_1_pos = 0.5 * P * tf.sqrt(1.0 + self.a_11 - self.a_22 - self.a_33)
        q_2_pos = 0.5 * P * tf.sqrt(1.0 - self.a_11 + self.a_22 - self.a_33)
        q_3_pos = 0.5 * P * tf.sqrt(1.0 - self.a_11 - self.a_22 + self.a_33)
        q_1 = tf.where(self.a_32 < self.a_23, -q_1_pos, q_1_pos)
        q_2 = tf.where(self.a_13 < self.a_31, -q_2_pos, q_2_pos)
        q_3 = tf.where(self.a_21 < self.a_12, -q_3_pos, q_3_pos)
        q_unnormalized = tf.concat([q_0, q_1, q_2, q_3], axis=-1)
        q_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(q_unnormalized), axis=-1, keepdims=True))
        return mcrpy.orientation.Quaternion(
            tf.where(
                q_norm > EPS,
                q_unnormalized / q_norm,
                tf.concat(
                    [
                        tf.ones(tf.shape(q_0), dtype=tf.float64),
                        tf.zeros(tf.shape(q_0), dtype=tf.float64),
                        tf.zeros(tf.shape(q_0), dtype=tf.float64),
                        tf.zeros(tf.shape(q_0), dtype=tf.float64),
                    ],
                    axis=-1,
                ),
            )
        )

    def __add__(self, other: Matrix):
        return Matrix(self.x + other.x)

    def __truediv__(self, other: int):
        return Matrix(self.x / tf.constant(other, dtype=tf.float64))

    @property
    def a_11(self):
        return tf.gather(self.x, [0], axis=-1)

    @property
    def a_12(self):
        return tf.gather(self.x, [1], axis=-1)

    @property
    def a_13(self):
        return tf.gather(self.x, [2], axis=-1)

    @property
    def a_21(self):
        return tf.gather(self.x, [3], axis=-1)

    @property
    def a_22(self):
        return tf.gather(self.x, [4], axis=-1)

    @property
    def a_23(self):
        return tf.gather(self.x, [5], axis=-1)

    @property
    def a_31(self):
        return tf.gather(self.x, [6], axis=-1)

    @property
    def a_32(self):
        return tf.gather(self.x, [7], axis=-1)

    @property
    def a_33(self):
        return tf.gather(self.x, [8], axis=-1)

    @property
    def is_unit_matrix(self):
        unit_tensor_ones = tf.gather(self.x, indices=[0, 4, 8], axis=-1)
        unit_tensor_zeros = tf.gather(self.x, indices=[1, 2, 3, 5, 6, 7], axis=-1)
        validated_unit_tensor_ones = tf.math.reduce_all(unit_tensor_ones < 1 + EPS) and tf.math.reduce_all(
            unit_tensor_ones > 1 - EPS
        )
        validated_unit_tensor_zeros = tf.math.reduce_all(unit_tensor_zeros < EPS) and tf.math.reduce_all(
            unit_tensor_zeros > -EPS
        )
        return validated_unit_tensor_ones and validated_unit_tensor_zeros

    @property
    def as_matrix(self):
        return self.to_matrix_shape()

    def to_matrix_shape(self):
        return tf.stack(
            [
                tf.concat([self.a_11, self.a_21, self.a_31], axis=-1),
                tf.concat([self.a_12, self.a_22, self.a_32], axis=-1),
                tf.concat([self.a_13, self.a_23, self.a_33], axis=-1),
            ],
            axis=-1,
        )

    @classmethod
    def from_matrix_shape(cls, array: tf.Tensor) -> Matrix:
        row_1 = tf.squeeze(tf.gather(array, [0], axis=-2), -2)
        row_2 = tf.squeeze(tf.gather(array, [1], axis=-2), -2)
        row_3 = tf.squeeze(tf.gather(array, [2], axis=-2), -2)
        a_11 = tf.gather(row_1, [0], axis=-1)
        a_12 = tf.gather(row_1, [1], axis=-1)
        a_13 = tf.gather(row_1, [2], axis=-1)
        a_21 = tf.gather(row_2, [0], axis=-1)
        a_22 = tf.gather(row_2, [1], axis=-1)
        a_23 = tf.gather(row_2, [2], axis=-1)
        a_31 = tf.gather(row_3, [0], axis=-1)
        a_32 = tf.gather(row_3, [1], axis=-1)
        a_33 = tf.gather(row_3, [2], axis=-1)
        return cls(tf.concat([a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32, a_33], axis=-1))

    @property
    def as_transposed_matrix(self):
        return tf.einsum("...ij->...ji", self.to_matrix_shape())

    def times(self, other: Matrix) -> Matrix:
        result = tf.einsum("...ij,...jk->...ik", self.to_matrix_shape(), other.to_matrix_shape())
        return Matrix.from_matrix_shape(result)

    def transposed_times(self, other: Matrix) -> Matrix:
        result = tf.einsum("...ij,...ik->...jk", self.to_matrix_shape(), other.to_matrix_shape())
        return Matrix.from_matrix_shape(result)

    def rotate(self, rotation: mcrpy.Orientation.Orientation) -> Orientation:
        return rotation.astype(Matrix).times(self)

    def validate(self):
        validated_shape = tf.shape(self.x)[-1] == 9
        return validated_shape


if __name__ == "__main__":
    import numpy as np

    m = Matrix(tf.constant(np.array([0, 1, 0, -1, 0, 0, 0, 0, 1]).astype(float), dtype=tf.float64))
    print(m)
    print(m.to_matrix_shape())
    print(Matrix.from_matrix_shape(m.to_matrix_shape()))
