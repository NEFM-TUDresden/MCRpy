from __future__ import annotations
import tensorflow as tf

from mcrpy.orientation.Quaternion import Quaternion
from mcrpy.orientation.Orientation import Orientation, P, EPS


class UnnormalizedQuaternion(Orientation):
    x: tf.Tensor
    n_dims: int = 4

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> UnnormalizedQuaternion:
        return UnnormalizedQuaternion(q.x)

    def to_quaternion(self) -> Quaternion:
        magnitude = self.magnitude
        return Quaternion(self.x / (tf.where(magnitude > EPS, magnitude, EPS)))

    @property
    def magnitude(self) -> tf.Tensor:
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(self.x), axis=-1, keepdims=True))

    def __validate__(self) -> bool:
        return self.validate()

    def validate(self) -> bool:
        return tf.shape(self.x)[-1] == 4
