"""Based on Rowenhorst et al 2015"""

from __future__ import annotations
import tensorflow as tf

from mcrpy.orientation.Orientation import Orientation, P, EPS


class Quaternion(Orientation):
    x: tf.Tensor
    n_dims: int = 4

    def times(self, other: Quaternion) -> Quaternion:
        versor_shape = tf.cond(
            tf.size(self.q_vec) > tf.size(other.q_vec), lambda: tf.shape(self.q_vec), lambda: tf.shape(other.q_vec)
        )
        scalar_part = self.q_0 * other.q_0 - tf.math.reduce_sum(self.q_vec * other.q_vec, axis=-1, keepdims=True)
        vector_part = (
            self.q_0 * other.q_vec
            + other.q_0 * self.q_vec
            + P * tf.linalg.cross(tf.broadcast_to(self.q_vec, versor_shape), tf.broadcast_to(other.q_vec, versor_shape))
        )
        return Quaternion(tf.concat([scalar_part, vector_part], axis=-1))

    def rotate(self, rotation: Orientation) -> Orientation:
        return rotation.to_quaternion().times(self)

    def rotate_vector(self, vector: tf.Tensor) -> tf.Tensor:
        tf.assert_equal(tf.size(vector), 3, message="only a single vector")
        v_1 = tf.gather(vector, [0], axis=-1)
        v_2 = tf.gather(vector, [1], axis=-1)
        v_3 = tf.gather(vector, [2], axis=-1)
        p_0 = tf.zeros(tf.shape(v_1), dtype=tf.float64)
        p = Quaternion(tf.concat([p_0, v_1, v_2, v_3], axis=-1))
        p_rot = self.times(p.times(self.conjugate))
        return tf.gather(p_rot.x, [1, 2, 3], axis=-1)

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> Quaternion:
        return q

    @classmethod
    def from_unnormalized_entries(cls, q_entries: tf.Tensor) -> Quaternion:
        q_mag = tf.math.sqrt(tf.reduce_sum(tf.math.square(q_entries), axis=-1, keepdims=True))
        return Quaternion(
            tf.where(
                q_mag < EPS,
                tf.concat(
                    [
                        tf.ones(tf.shape(q_mag), dtype=tf.float64),
                        tf.zeros(tf.shape(q_mag), dtype=tf.float64),
                        tf.zeros(tf.shape(q_mag), dtype=tf.float64),
                        tf.zeros(tf.shape(q_mag), dtype=tf.float64),
                    ],
                    axis=-1,
                ),
                q_entries / q_mag,
            )
        )

    def to_quaternion(self) -> Quaternion:
        return self

    def to_axis(self):
        from mcrpy.orientation.Axis import Axis

        return Axis.from_quaternion(self)

    @property
    def q_0(self) -> tf.Tensor:
        return tf.gather(self.x, [0], axis=-1)

    @property
    def q_1(self) -> tf.Tensor:
        return tf.gather(self.x, [1], axis=-1)

    @property
    def q_2(self) -> tf.Tensor:
        return tf.gather(self.x, [2], axis=-1)

    @property
    def q_3(self) -> tf.Tensor:
        return tf.gather(self.x, [3], axis=-1)

    @property
    def q_vec(self) -> tf.Tensor:
        return tf.gather(self.x, [1, 2, 3], axis=-1)

    @property
    def conjugate(self) -> Quaternion:
        return Quaternion(tf.concat([self.q_0, -self.q_vec], axis=-1))

    def validate(self) -> bool:
        return tf.shape(self.x)[-1] == 4

    def __validate__(self):
        tf.assert_equal(self.validate(), True)

    def dot(self, other: Quaternion) -> tf.Tensor:
        return tf.reduce_sum(self.x * other.x, axis=-1)

    def average(self, other: Quaternion) -> Quaternion:
        return Quaternion.from_unnormalized_entries(self.x + other.x)
