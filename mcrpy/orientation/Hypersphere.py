"""Based on Rowenhorst et al 2015"""

from __future__ import annotations
import numpy as np
import tensorflow as tf

import mcrpy.orientation.Quaternion
from mcrpy.orientation.Orientation import Orientation, P, EPS


class Hypersphere(Orientation):
    x: tf.Tensor
    n_dims: int = 3

    @classmethod
    def from_quaternion(cls, q: mcrpy.orientation.Quaternion) -> Hypersphere:
        return cls(quaternion2hypersphere(q.x))

    def to_quaternion(self) -> mcrpy.orientation.Quaternion:
        omega_half = 0.5 * self.omega
        q_0 = tf.cos(omega_half)
        sin = tf.sin(omega_half)
        q_1 = sin * tf.sin(self.theta) * tf.cos(self.phi)
        q_2 = sin * tf.sin(self.theta) * tf.sin(self.phi)
        q_3 = sin * tf.cos(self.theta)
        return mcrpy.orientation.Quaternion(tf.concat([q_0, q_1, q_2, q_3], axis=-1))

    @property
    def omega(self) -> tf.Tensor:
        return tf.gather(self.x, indices=[0], axis=-1)

    @property
    def theta(self) -> tf.Tensor:
        return tf.gather(self.x, indices=[1], axis=-1)

    @property
    def phi(self) -> tf.Tensor:
        return tf.gather(self.x, indices=[2], axis=-1)

    def __validate__(self) -> bool:
        return tf.shape(self.x)[-1] == 3

    def validate(self) -> bool:
        validated_shape = tf.shape(self.x)[-1] == 3
        validated_omega = tf.logical_and(
            tf.reduce_all(self.omega >= 0.0),
            tf.reduce_all(2 * np.pi >= self.omega),
        )
        validated_theta = tf.logical_and(tf.reduce_all(self.theta >= 0.0), tf.reduce_all(np.pi >= self.theta))
        validated_phi = tf.logical_and(tf.reduce_all(self.phi >= 0.0), tf.reduce_all(2 * np.pi >= self.phi))
        validated_magnitude = tf.logical_and(tf.logical_and(validated_omega, validated_theta), validated_phi)
        return tf.logical_and(validated_shape, validated_magnitude)


def _quaternion2hypersphere_case1(q_0: tf.Tensor):  # q_0 = 1
    omega = tf.zeros(tf.shape(q_0), dtype=tf.float64)
    theta = tf.zeros(tf.shape(q_0), dtype=tf.float64)
    phi = tf.zeros(tf.shape(q_0), dtype=tf.float64)
    return tf.concat([omega, theta, phi], axis=-1)


def _quaternion2hypersphere_case2(q: tf.Tensor):
    q_0 = tf.gather(q, [0], axis=-1)
    q_1 = tf.gather(q, [1], axis=-1)
    q_2 = tf.gather(q, [2], axis=-1)
    q_3 = tf.gather(q, [3], axis=-1)
    q_vec_norm = tf.math.sqrt(tf.math.square(q_1) + tf.math.square(q_2) + tf.math.square(q_3))
    x = q_1 / q_vec_norm
    y = q_2 / q_vec_norm
    z = q_3 / q_vec_norm
    omega = 2.0 * tf.acos(q_0)
    theta = tf.acos(z)
    phi = tf.where(theta == 0.0, tf.zeros(tf.shape(q_0), dtype=tf.float64), _quaternion2hypersphere_case3(x, y))
    return tf.concat([omega, theta, phi], axis=-1)


def _quaternion2hypersphere_case3(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    phi = tf.atan2(y, x)
    return tf.where(phi < 0.0, 2.0 * np.pi + phi, phi)


def quaternion2hypersphere(ori_quaternion: tf.Tensor) -> tf.Tensor:
    q_0 = tf.gather(ori_quaternion, [0], axis=-1)
    return tf.where(q_0 == 1.0, _quaternion2hypersphere_case1(q_0), _quaternion2hypersphere_case2(ori_quaternion))
