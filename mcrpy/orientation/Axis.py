"""Based on Rowenhorst et al 2015"""

from __future__ import annotations
import numpy as np
import tensorflow as tf

from mcrpy.orientation.Rodrigues import Rodrigues
from mcrpy.orientation.Quaternion import Quaternion
from mcrpy.orientation.Orientation import Orientation

EPS = 1e-5


class Axis(Orientation):
    x: tf.Tensor
    n_dims: int = 4

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> Axis:
        angle = 2.0 * tf.acos(q.q_0)
        return cls(
            tf.where(
                tf.abs(q.q_0) > 1.0 - 1e-15,
                _quaternion2axis_case1(q.q_0),
                tf.where(
                    tf.abs(q.q_0) < 1e-15,
                    _quaternion2axis_case2(q.q_1, q.q_2, q.q_3),
                    _quaternion2axis_case3(q.q_0, q.q_1, q.q_2, q.q_3, angle),
                ),
            )
        )

    def to_quaternion(self) -> Quaternion:
        omega_half = self.angle * 0.5
        q_0 = tf.cos(omega_half)
        q_vec = tf.sin(omega_half) * self.axis
        return Quaternion(tf.concat([q_0, q_vec], axis=-1))

    def to_rodrigues(self) -> Rodrigues:
        return Rodrigues(self.axis * tf.tan(0.5 * self.angle))

    @property
    def angle(self) -> tf.Tensor:
        return tf.gather(self.x, [3], axis=-1)

    @property
    def axis(self) -> tf.Tensor:
        return tf.gather(self.x, [0, 1, 2], axis=-1)

    def validate(self) -> bool:
        validated_shape = tf.shape(self.x)[-1] == 4
        magnitudes = tf.math.reduce_sum(tf.math.square(self.axis), axis=-1)
        validated_magnitude = tf.logical_and(
            tf.math.reduce_all(magnitudes < 1 + EPS), tf.math.reduce_all(magnitudes > 1 - EPS)
        )
        return tf.logical_and(validated_shape, validated_magnitude)

    def __validate__(self):
        tf.assert_equal(self.validate(), True)

    def astype(self, orientation_type: type) -> Orientation:
        if orientation_type == Rodrigues:
            return self.to_rodrigues()
        return super().astype(orientation_type)


def _quaternion2axis_case1(q_0: tf.Tensor) -> tf.Tensor:
    n_1 = tf.zeros(tf.shape(q_0), dtype=tf.float64)
    n_2 = tf.zeros(tf.shape(q_0), dtype=tf.float64)
    n_3 = tf.ones(tf.shape(q_0), dtype=tf.float64)
    angle = tf.zeros(tf.shape(q_0), dtype=tf.float64)
    return tf.concat([n_1, n_2, n_3, angle], axis=-1)


def _quaternion2axis_case2(q_1: tf.Tensor, q_2: tf.Tensor, q_3: tf.Tensor) -> tf.Tensor:
    angle = tf.ones(tf.shape(q_1), dtype=tf.float64) * np.pi
    return tf.concat([q_1, q_2, q_3, angle], axis=-1)


def _quaternion2axis_case3(
    q_0: tf.Tensor, q_1: tf.Tensor, q_2: tf.Tensor, q_3: tf.Tensor, angle: tf.Tensor
) -> tf.Tensor:
    s = tf.sign(q_0) / tf.sqrt(tf.square(q_1) + tf.square(q_2) + tf.square(q_3))
    return tf.concat([s * q_1, s * q_2, s * q_3, tf.where(q_0 < 0.0, 2 * np.pi - angle, angle)], axis=-1)


def test_axis_validation():
    from mcrpy.orientation.test_orientation import _generate_axis_testdata
    from mcrpy.orientation.Rodrigues import Rodrigues

    data = Quaternion(tf.constant(np.array([0.999999999999999, 0, 0, 0], dtype=float), dtype=tf.float64)).astype(Axis)
    data = data.to_quaternion().astype(Rodrigues)
    data = data.astype(Axis)
    print(data)


if __name__ == "__main__":
    test_axis_validation()
