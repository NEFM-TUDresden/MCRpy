"""Based on Rowenhorst et al 2015"""

from __future__ import annotations
import tensorflow as tf

import mcrpy.orientation.Axis
import mcrpy.orientation.Quaternion
from mcrpy.orientation.Orientation import Orientation, P, EPS


class Rodrigues(Orientation):
    x: tf.Tensor
    n_dims: int = 3

    @classmethod
    def from_quaternion(cls, q: mcrpy.orientation.Quaternion) -> Rodrigues:
        return mcrpy.orientation.Axis.from_quaternion(q).to_rodrigues()

    def to_axis(self) -> mcrpy.orientation.Axis:
        magnitude = self.magnitude
        return mcrpy.orientation.Axis(
            tf.where(
                tf.math.abs(magnitude) > EPS,
                tf.concat([self.axis, self.angle], axis=-1),
                tf.concat(
                    [
                        tf.zeros(tf.shape(magnitude), dtype=tf.float64),
                        tf.zeros(tf.shape(magnitude), dtype=tf.float64),
                        tf.ones(tf.shape(magnitude), dtype=tf.float64),
                        tf.zeros(tf.shape(magnitude), dtype=tf.float64),
                    ],
                    axis=-1,
                ),
            )
        )

    def to_quaternion(self) -> mcrpy.orientation.Quaternion:
        return self.to_axis().to_quaternion()

    def times(self, other: Rodrigues) -> Rodrigues:
        versor_shape = tf.cond(tf.size(self.x) > tf.size(other.x), lambda: tf.shape(self.x), lambda: tf.shape(other.x))
        denominator = 1.0 - tf.math.reduce_sum(self.x * other.x, axis=-1, keepdims=True)
        numerator = (
            self.x
            + other.x
            + P * tf.linalg.cross(tf.broadcast_to(self.x, versor_shape), tf.broadcast_to(other.x, versor_shape))
        )
        return Rodrigues(tf.math.divide_no_nan(numerator, denominator))

    def rotate(self, rotation: Orientation) -> Orientation:
        return rotation.astype(Rodrigues).times(self)

    @property
    def magnitude(self) -> tf.Tensor:
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(self.x), axis=-1, keepdims=True))

    @property
    def axis(self) -> tf.Tensor:
        return self.x / self.magnitude

    @property
    def angle(self) -> tf.Tensor:
        return 2.0 * tf.math.atan(self.magnitude)

    def validate(self) -> bool:
        return tf.shape(self.x)[-1] == 3

    def astype(self, orientation_type: type) -> Orientation:
        if orientation_type == mcrpy.orientation.Axis:
            return self.to_axis()
        return super().astype(orientation_type)


def test_times():
    from mcrpy.orientation.test_orientation import _generate_axis_testdata

    data = _generate_axis_testdata([100, 100, 100]).to_quaternion()
    test_rotation = _generate_axis_testdata([1])
    rot_1 = data.rotate(test_rotation)
    rot_2 = data.astype(Rodrigues).rotate(test_rotation).to_quaternion()
    from mcrpy.orientation.Axis import Axis

    deviation = rot_1.rotate(rot_2.conjugate).astype(Axis).angle
    assert tf.reduce_max(tf.abs(deviation)) < 1e-5


if __name__ == "__main__":
    test_times()
