"""Based on Rowenhorst et al 2015"""

from __future__ import annotations
import numpy as np
import tensorflow as tf

import mcrpy.orientation.Quaternion
from mcrpy.orientation.Orientation import Orientation, P, EPS


class Euler(Orientation):
    x: tf.Tensor
    n_dims: int = 3

    @classmethod
    def from_quaternion(cls, q: mcrpy.orientation.Quaternion) -> Euler:
        return cls(quaternion2euler(q.x))

    def to_quaternion(self) -> mcrpy.orientation.Quaternion:
        phi_1 = self.phi_1
        PHI = self.Phi
        phi_2 = self.phi_2
        sigma = 0.5 * (phi_1 + phi_2)
        delta = 0.5 * (phi_1 - phi_2)
        c = tf.cos(0.5 * PHI)
        s = tf.sin(0.5 * PHI)
        q_0 = c * tf.cos(sigma)
        q_1 = -1.0 * P * s * tf.cos(delta)
        q_2 = -1.0 * P * s * tf.sin(delta)
        q_3 = -1.0 * P * c * tf.sin(sigma)
        q = tf.concat([q_0, q_1, q_2, q_3], axis=-1)
        return mcrpy.orientation.Quaternion(tf.where(q_0 >= 0.0, q, -1.0 * q))

    @property
    def phi_1(self):
        return tf.gather(self.x, [0], axis=-1)

    @property
    def Phi(self):
        return tf.gather(self.x, [1], axis=-1)

    @property
    def phi_2(self):
        return tf.gather(self.x, [2], axis=-1)

    def validate(self):
        return tf.shape(self.x)[-1] == 3


def _quaternion2euler_case1(q: tf.Tensor, q_03: tf.Tensor, q_12: tf.Tensor, chi: tf.Tensor) -> tf.Tensor:  # A22
    q_0 = tf.gather(q, [0], axis=-1)
    q_1 = tf.gather(q, [1], axis=-1)
    q_2 = tf.gather(q, [2], axis=-1)
    q_3 = tf.gather(q, [3], axis=-1)
    phi_1 = tf.atan2((q_1 * q_3 - P * q_0 * q_2) / chi, (-1.0 * P * q_0 * q_1 - q_2 * q_3) / chi)
    phi_2 = tf.atan2((P * q_0 * q_2 + q_1 * q_3) / chi, (q_2 * q_3 - P * q_0 * q_1) / chi)
    PHI = tf.atan2(2 * chi, q_03 - q_12)
    return tf.concat([phi_1, PHI, phi_2], axis=-1)


def _quaternion2euler_case2(q: tf.Tensor) -> tf.Tensor:  # A20
    q_0 = tf.gather(q, [0], axis=-1)
    q_3 = tf.gather(q, [3], axis=-1)
    phi_1 = tf.atan2(-2.0 * P * q_0 * q_3, tf.math.square(q_0) - tf.math.square(q_3))
    phi_2 = tf.zeros(tf.shape(phi_1), dtype=tf.float64)
    PHI = tf.zeros(tf.shape(phi_1), dtype=tf.float64)
    return tf.concat([phi_1, PHI, phi_2], axis=-1)


def _quaternion2euler_case3(q: tf.Tensor) -> tf.Tensor:  # A21
    q_1 = tf.gather(q, [1], axis=-1)
    q_2 = tf.gather(q, [2], axis=-1)
    phi_1 = tf.atan2(2.0 * q_1 * q_2, tf.math.square(q_1) - tf.math.square(q_2))
    phi_2 = tf.zeros(tf.shape(phi_1), dtype=tf.float64)
    PHI = tf.cast(tf.ones(tf.shape(phi_1), dtype=tf.float64) * np.pi, tf.float64)
    return tf.concat([phi_1, PHI, phi_2], axis=-1)


def quaternion2euler(ori_quaternion: tf.Tensor) -> tf.Tensor:
    q_03 = tf.reduce_sum(tf.math.square(tf.gather(ori_quaternion, indices=[0, 3], axis=-1)), axis=-1, keepdims=True)
    q_12 = tf.reduce_sum(tf.math.square(tf.gather(ori_quaternion, indices=[1, 2], axis=-1)), axis=-1, keepdims=True)
    chi = tf.sqrt(q_03 * q_12)
    return tf.where(
        chi != 0.0,
        x=_quaternion2euler_case1(ori_quaternion, q_03, q_12, chi),
        y=tf.where(q_12 == 0, x=_quaternion2euler_case2(ori_quaternion), y=_quaternion2euler_case3(ori_quaternion)),
    )
