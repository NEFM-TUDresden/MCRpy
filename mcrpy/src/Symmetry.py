from typing import Dict, List, Union, Tuple
import logging

import numpy as np
import tensorflow as tf

from mcrpy.orientation import Axis, Orientation, Quaternion, Rodrigues, Matrix, Euler
from mcrpy.orientation.test_orientation import _generate_axis_testdata
from mcrpy.src.translation import translate_2D


class LimitingPlane:  # in rodrigues space
    def __init__(self, normal_vector: np.ndarray, position: float):
        assert normal_vector.size == 3 and len(normal_vector.shape) == 1
        assert 0.9999 < np.linalg.norm(normal_vector) < 1.0001
        self.normal_vector = tf.constant(normal_vector, dtype=tf.float64)
        self.position = tf.constant(position, dtype=tf.float64)
        self.rho_mag = tf.constant(np.tan(2.0 * np.arctan(position)), dtype=tf.float64)

    def to_rodrigues(self) -> Rodrigues:
        return Rodrigues(self.normal_vector * self.rho_mag)


ONE = tf.constant(1.0, dtype=tf.float64)
ZERO = tf.constant(0.0, dtype=tf.float64)
EPS = tf.constant(1.0e-15, dtype=tf.float64)


class Symmetry:
    def __init__(
        self,
        symmetry_operations: List[Orientation],
        rgb_vectors: List[List[int]],
        fz_limiting_planes: List[LimitingPlane],
        expansion_table: Dict[Tuple[int, int], List[Tuple[int, int, int, bool]]] = None,
    ):
        self.symmetry_operations = [so.to_quaternion() for so in symmetry_operations]
        self.standard_triangle = np.linalg.inv(np.array([self._normalize(e_i) for e_i in rgb_vectors]).T)
        self.fz_limiting_planes = fz_limiting_planes
        self.expansion_table = expansion_table

    @staticmethod
    def _normalize(a: np.ndarray):
        a_norm = np.linalg.norm(a)
        if a_norm == 0.0:
            raise ValueError
        return a / a_norm

    @property
    def n_symmetries(self):
        return len(self.symmetry_operations)

    def in_standard_triangle(self, vectors: tf.Tensor):
        tf.assert_equal(tf.shape(vectors)[-1], 3)
        return tf.reduce_all(tf.einsum("ij,...j", self.standard_triangle, vectors) >= -1e-12, axis=-1, keepdims=True)

    def apply_to_orientation(self, ori: Orientation) -> List[Orientation]:
        return [ori.rotate(symmetry) for symmetry in self.symmetry_operations]

    def project_to_fz(self, ori: Orientation) -> Orientation:
        return type(ori)(tf.where(tf.expand_dims(self.in_fz(ori), axis=-1), ori.x, self.project_all_to_fz(ori).x))

    @tf.function
    def get_closest_equivalents_parallel(self, ori: Orientation, ori_ref: Orientation) -> Orientation:
        og_type = type(ori)
        ori_via = ori.to_quaternion()
        ori_ref = ori_ref.to_quaternion()
        q_0 = tf.gather(ori_via.x, 0, axis=-1)
        q_1 = tf.gather(ori_via.x, 1, axis=-1)
        q_2 = tf.gather(ori_via.x, 2, axis=-1)
        q_3 = tf.gather(ori_via.x, 3, axis=-1)
        symmetric_equivalents_q = self.apply_to_orientation(ori_via)
        dot_products = [ori_ref.dot(seq_i) for seq_i in symmetric_equivalents_q]
        symmetric_maxima = tf.math.argmin(tf.stack(dot_products, axis=0), axis=0)
        symmetric_equivalents = tf.concat([tf.expand_dims(i.x, axis=0) for i in symmetric_equivalents_q], axis=0)
        symmetric_q_0 = tf.gather(symmetric_equivalents, 0, axis=-1)
        symmetric_q_1 = tf.gather(symmetric_equivalents, 1, axis=-1)
        symmetric_q_2 = tf.gather(symmetric_equivalents, 2, axis=-1)
        symmetric_q_3 = tf.gather(symmetric_equivalents, 3, axis=-1)
        for update_symmetry_nr in range(self.n_symmetries):
            q_0 = tf.where(symmetric_maxima == update_symmetry_nr, symmetric_q_0[update_symmetry_nr], q_0)
            q_1 = tf.where(symmetric_maxima == update_symmetry_nr, symmetric_q_1[update_symmetry_nr], q_1)
            q_2 = tf.where(symmetric_maxima == update_symmetry_nr, symmetric_q_2[update_symmetry_nr], q_2)
            q_3 = tf.where(symmetric_maxima == update_symmetry_nr, symmetric_q_3[update_symmetry_nr], q_3)
        ori_fz = tf.stack([q_0, q_1, q_2, q_3], axis=-1)  # [0]
        ori_fz = tf.reshape(ori_fz, tf.shape(ori_via.x))
        return Quaternion(ori_fz).astype(og_type)

    def get_closest_equivalents(self, ori: Orientation, ori_ref: Orientation) -> Orientation:
        og_type = type(ori)
        ori_via = ori.to_quaternion()
        ori_via_cp = Quaternion(tf.identity(ori_via.x))
        ori_ref = Quaternion(tf.reshape(ori_ref.to_quaternion().x, tf.shape(ori_via.x)))
        q_0 = ori_via.q_0
        q_1 = ori_via.q_1
        q_2 = ori_via.q_2
        q_3 = ori_via.q_3
        for symmetry in self.symmetry_operations:
            symmetric_equivalent = ori_via_cp.rotate(symmetry)
            smaller_rotation = ori_ref.dot(symmetric_equivalent) > ori_ref.dot(ori_via)
            smaller_rotation = tf.reshape(smaller_rotation, tf.shape(q_0))
            q_0 = tf.where(smaller_rotation, symmetric_equivalent.q_0, q_0)
            q_1 = tf.where(smaller_rotation, symmetric_equivalent.q_1, q_1)
            q_2 = tf.where(smaller_rotation, symmetric_equivalent.q_2, q_2)
            q_3 = tf.where(smaller_rotation, symmetric_equivalent.q_3, q_3)
            ori_via = tf.stack([q_0, q_1, q_2, q_3], axis=-1)
            ori_via = Quaternion(tf.reshape(ori_via, tf.shape(ori_via_cp.x)))
        return ori_via.astype(og_type)

    def compute_grain_boundaries(self, ori: Orientation = None) -> tf.Tensor:
        gb_list = []
        print("TODO ori should be 2D")
        for translation_x, translation_y in [(1, 0), (0, 1), (1, 1)]:
            q_translated = type(ori)(
                tf.reshape(translate_2D(ori.x, translation_x, translation_y)[0], tf.shape(ori.x))
            ).to_quaternion()
            q_translated = self.get_closest_equivalents_parallel(q_translated, ori)
            ori_delta = ori.to_quaternion().rotate(q_translated.conjugate).to_axis().angle
            gb_list.append(ori_delta / 3.14)
        return tf.concat(gb_list, axis=-1)

    @tf.function
    def compute_misorientation(self, ori_1: Orientation = None, ori: Orientation = None) -> tf.Tensor:
        ori_closest = self.get_closest_equivalents(ori_1, ori).to_quaternion()
        ori_delta = ori_closest.conjugate.rotate(ori)
        ori_delta_fz = self.project_all_to_fz_parallel(ori_delta)
        return ori_delta_fz.to_axis().angle

    def project_all_to_fz_parallel(self, ori: Orientation) -> Orientation:
        og_type = type(ori)
        ori_via = ori.astype(Quaternion)
        q_0 = ori_via.q_0
        q_1 = ori_via.q_1
        q_2 = ori_via.q_2
        q_3 = ori_via.q_3
        symmetric_equivalents = tf.concat(
            [tf.expand_dims(i.x, axis=0) for i in self.apply_to_orientation(ori_via)], axis=0
        )
        symmetric_q_0 = tf.gather(symmetric_equivalents, [0], axis=-1)
        symmetric_q_1 = tf.gather(symmetric_equivalents, [1], axis=-1)
        symmetric_q_2 = tf.gather(symmetric_equivalents, [2], axis=-1)
        symmetric_q_3 = tf.gather(symmetric_equivalents, [3], axis=-1)
        symmetric_maxima = tf.expand_dims(tf.math.argmax(tf.math.abs(symmetric_q_0), axis=0), 0)
        for update_symmetry_nr in range(self.n_symmetries):
            q_0 = tf.where(symmetric_maxima == update_symmetry_nr, symmetric_q_0[update_symmetry_nr], q_0)
            q_1 = tf.where(symmetric_maxima == update_symmetry_nr, symmetric_q_1[update_symmetry_nr], q_1)
            q_2 = tf.where(symmetric_maxima == update_symmetry_nr, symmetric_q_2[update_symmetry_nr], q_2)
            q_3 = tf.where(symmetric_maxima == update_symmetry_nr, symmetric_q_3[update_symmetry_nr], q_3)
        ori_fz = tf.concat([q_0, q_1, q_2, q_3], axis=-1)[0]
        ori_fz = tf.reshape(ori_fz, tf.shape(ori_via.x))
        return Quaternion(ori_fz).astype(og_type)

    def project_all_to_fz_naive(self, ori: Orientation) -> Orientation:
        og_type = type(ori)
        ori_via = ori.astype(Quaternion)
        ori_via_cp = Quaternion(tf.identity(ori_via.x))
        q_0 = ori_via.q_0
        q_1 = ori_via.q_1
        q_2 = ori_via.q_2
        q_3 = ori_via.q_3
        for symmetry in self.symmetry_operations:
            symmetric_equivalent = ori_via_cp.rotate(symmetry).x
            symmetric_q_0 = tf.gather(symmetric_equivalent, [0], axis=-1)
            symmetric_q_1 = tf.gather(symmetric_equivalent, [1], axis=-1)
            symmetric_q_2 = tf.gather(symmetric_equivalent, [2], axis=-1)
            symmetric_q_3 = tf.gather(symmetric_equivalent, [3], axis=-1)
            smaller_rotation = tf.abs(symmetric_q_0) > tf.abs(q_0)
            q_0 = tf.where(smaller_rotation, symmetric_q_0, q_0)
            q_1 = tf.where(smaller_rotation, symmetric_q_1, q_1)
            q_2 = tf.where(smaller_rotation, symmetric_q_2, q_2)
            q_3 = tf.where(smaller_rotation, symmetric_q_3, q_3)
        ori_fz = tf.concat([q_0, q_1, q_2, q_3], axis=-1)
        ori_fz = tf.reshape(ori_fz, tf.shape(ori_via.x))
        return Quaternion(ori_fz).astype(og_type)

    def _project_single_pass(self, ori_via: Rodrigues) -> Rodrigues:
        for fzlp in self.fz_limiting_planes:
            abs_rodrigues = tf.math.abs(ori_via.x)
            plane_beyond_ori = fzlp.position >= tf.math.reduce_sum(fzlp.normal_vector * abs_rodrigues, axis=-1)
            if tf.reduce_all(plane_beyond_ori):
                continue
            r_fzlp = Rodrigues(-1.0 * fzlp.rho_mag * fzlp.normal_vector * tf.sign(ori_via.x))
            ori_via_x = tf.where(
                tf.broadcast_to(tf.expand_dims(plane_beyond_ori, -1), tf.shape(ori_via.x)),
                ori_via.x,
                tf.reshape(ori_via.rotate(r_fzlp).x, tf.shape(ori_via.x)),
            )
            ori_via = Rodrigues(ori_via_x)
        return ori_via

    def project_all_to_fz(self, ori: Orientation) -> Orientation:
        og_type = type(ori)
        ori_via = ori.astype(Rodrigues)
        while not self.all_in_fz(ori_via):
            ori_via = self._project_single_pass(ori_via)
        return ori_via.astype(og_type)

    def all_in_fz(self, ori: Orientation, EPS: float = 0.0) -> bool:
        return tf.reduce_all(self.in_fz(ori, EPS=EPS))

    def project_to_standard_triangle_parallel(
        self,
        ori: Orientation,  # orientations to rotate vector
        vector: tf.Tensor = tf.constant([0, 0, 1], dtype=tf.float64),  # vector to align with
    ):
        symmetric_equivalents = self.apply_to_orientation(ori)
        equivalent_vectors = [se.rotate_vector(vector) for se in symmetric_equivalents]
        equivalent_vectors = [
            equivalent_vector / tf.sqrt(tf.reduce_sum(tf.square(equivalent_vector)))
            for equivalent_vector in equivalent_vectors
        ]
        equivalent_vectors = tf.stack(equivalent_vectors, axis=0)
        in_st = self.in_standard_triangle(equivalent_vectors)
        equivalent_vectors = tf.where(in_st, equivalent_vectors, -equivalent_vectors)
        in_st = self.in_standard_triangle(equivalent_vectors)
        tf.assert_greater(tf.reduce_min(tf.reduce_sum(tf.cast(in_st, tf.int32), axis=0)), 0)
        vectors_in_triangle = tf.zeros(tf.shape(equivalent_vectors)[1:], dtype=tf.float64)
        for update_symmetry_nr in range(self.n_symmetries):
            vectors_in_triangle = tf.where(
                in_st[update_symmetry_nr], equivalent_vectors[update_symmetry_nr], vectors_in_triangle
            )
        return vectors_in_triangle

    def project_to_standard_triangle(
        self,
        ori: Orientation,  # orientations to rotate vector
        vector: tf.Tensor = tf.constant([0, 0, 1], dtype=tf.float64),  # vector to align with
    ):
        vectors_in_triangle = None
        ori_via = ori.to_quaternion()
        for symmetry in self.symmetry_operations:
            symmetric_equivalent = ori_via.rotate(symmetry)
            equivalent_vectors = symmetric_equivalent.rotate_vector(vector)
            equivalent_vectors = equivalent_vectors / tf.sqrt(
                tf.reduce_sum(tf.square(equivalent_vectors), axis=-1, keepdims=True)
            )
            in_st = self.in_standard_triangle(equivalent_vectors)
            equivalent_vectors = tf.where(in_st, equivalent_vectors, -equivalent_vectors)
            in_st = self.in_standard_triangle(equivalent_vectors)
            if vectors_in_triangle is not None:
                vectors_in_triangle = tf.where(in_st, equivalent_vectors, vectors_in_triangle)
            else:
                vectors_in_triangle = equivalent_vectors
        tf.assert_equal(tf.reduce_all(self.in_standard_triangle(vectors_in_triangle)), True)
        return vectors_in_triangle

    def to_ipf(self, ori: Orientation, vector: tf.Tensor = tf.constant([0, 0, 1], dtype=tf.float64)) -> tf.Tensor:
        vs = self.project_to_standard_triangle(ori, vector)
        rgb = tf.einsum("ij,...j", self.standard_triangle, vs)
        rgb = tf.math.sqrt(rgb)
        rgb_norm = tf.reduce_max(rgb, axis=-1, keepdims=True)
        rgb = tf.where(rgb_norm < 1e-9, ONE, rgb / rgb_norm)
        return rgb

    def in_fz(self, ori: Orientation, EPS: float = 0.0) -> bool:
        ori_rodrigues = ori.astype(Rodrigues)
        abs_rodrigues = tf.math.abs(ori_rodrigues.x)
        conditions = [
            fzlp.position + EPS >= tf.math.reduce_sum(fzlp.normal_vector * abs_rodrigues, axis=-1)
            for fzlp in self.fz_limiting_planes
        ]
        return tf.reduce_all(tf.stack(conditions, axis=0), axis=0)


Cubic = Symmetry(
    [
        Quaternion(np.array(q, dtype=np.float64))
        for q in [
            [1.0, 0.0, 0.0, 0.0],  # angle axis 0 0 0 0
            [0.0, 1.0, 0.0, 0.0],  # angle axis 180 1 0 0
            [0.0, 0.0, 1.0, 0.0],  # angle axis 180 0 1 0
            [0.0, 0.0, 0.0, 1.0],  # angle axis 180 0 0 1
            [0.0, 0.0, 0.5 * np.sqrt(2), 0.5 * np.sqrt(2)],  # angle axis 180 0 1 1 norm
            [0.0, 0.0, 0.5 * np.sqrt(2), -0.5 * np.sqrt(2)],  # angle axis 180 0 1 -1 norm
            [0.0, 0.5 * np.sqrt(2), 0.0, 0.5 * np.sqrt(2)],  # angle axis 180 1 0 1 norm
            [0.0, 0.5 * np.sqrt(2), 0.0, -0.5 * np.sqrt(2)],  # angle axis 180 1 0 -1 norm
            [0.0, 0.5 * np.sqrt(2), -0.5 * np.sqrt(2), 0.0],  # angle axis 180 1 -1 0 norm
            [0.0, -0.5 * np.sqrt(2), -0.5 * np.sqrt(2), 0.0],  # angle axis 180 -1 -1 0 norm
            [0.5, 0.5, 0.5, 0.5],  # angle axis 120 1 1 1 n
            [-0.5, 0.5, 0.5, 0.5],  # angle axis -120 1 1 1 n
            [-0.5, 0.5, 0.5, -0.5],  # angle axis -120 1 1 -1 n
            [-0.5, 0.5, -0.5, 0.5],  # angle axis -120 1 -1 1 n
            [-0.5, -0.5, 0.5, 0.5],  # angle axis -120 -1 1 1 n
            [-0.5, -0.5, 0.5, -0.5],  # angle axis -120 -1 1 -1 n
            [-0.5, -0.5, -0.5, 0.5],  # angle axis -120 -1 -1 1 n
            [-0.5, 0.5, -0.5, -0.5],  # angle axis -120 1 -1 -1 n
            [-0.5 * np.sqrt(2), 0.0, 0.0, 0.5 * np.sqrt(2)],  # angle axis -90 0 0 1
            [0.5 * np.sqrt(2), 0.0, 0.0, 0.5 * np.sqrt(2)],  # angle axis 90 0 0 1
            [-0.5 * np.sqrt(2), 0.0, 0.5 * np.sqrt(2), 0.0],  # angle axis -90 0 1 0
            [-0.5 * np.sqrt(2), 0.0, -0.5 * np.sqrt(2), 0.0],  # angle axis -90 0 1 0
            [-0.5 * np.sqrt(2), 0.5 * np.sqrt(2), 0.0, 0.0],  # angle axis -90 1 0 0
            [-0.5 * np.sqrt(2), -0.5 * np.sqrt(2), 0.0, 0.0],  # angle axis -90 1 0 0
        ]
    ],
    rgb_vectors=[
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    fz_limiting_planes=[
        LimitingPlane(n, p)
        for n, p in [
            (np.array([0, 0, 1]), np.tan(np.pi / 8)),
            (np.array([0, 1, 0]), np.tan(np.pi / 8)),
            (np.array([1, 0, 0]), np.tan(np.pi / 8)),
            (np.array([1, 1, 1]) / np.sqrt(3), np.tan(np.pi / 6)),
        ]
    ],
    expansion_table={
        (8, 1): [
            (0.333333333333, 0, 0, False),
            (0.308516766453, 4, 4, False),
            (0.365041960952, 4, 0, False),
            (-0.594588390011, 6, 4, False),
            (0.224733287488, 6, 0, False),
            (0.294627825494, 8, 8, False),
            (0.193373117038, 8, 4, False),
            (0.363609046177, 8, 0, False),
        ],
        (8, 2): [
            (0.333333333333, 1, 0, False),
            (-0.287213478952, 3, 0, False),
            (0.238976059700, 4, 4, True),
            (0.423659272868, 5, 4, False),
            (0.119352479007, 5, 0, False),
            (-0.460566186472, 6, 4, True),
            (-0.334932063529, 7, 4, False),
            (-0.022036911889, 7, 0, False),
            (0.456435464588, 8, 8, True),
            (0.149786172379, 8, 4, True),
        ],
        (8, 3): [
            (0.333333333333, 1, 1, False),
            (-0.227062192047, 3, 3, False),
            (0.175881617670, 3, 1, False),
            (-0.084490796177, 4, 3, True),
            (-0.223541634759, 4, 1, True),
            (0.167466031764, 5, 5, False),
            (0.274607982695, 5, 3, False),
            (0.300462606289, 5, 1, False),
            (0.270030862434, 6, 5, True),
            (0.315328111943, 6, 3, True),
            (-0.199431008804, 6, 1, True),
            (-0.114108866147, 7, 7, False),
            (-0.251199047646, 7, 5, False),
            (0.025246454204, 7, 3, False),
            (0.189488612701, 7, 1, False),
            (-0.114108866147, 8, 7, True),
            (-0.135015431217, 8, 5, True),
            (-0.145029837779, 8, 3, True),
            (-0.422475534112, 8, 1, True),
        ],
        (8, 4): [
            (0.333333333333, 1, 1, True),
            (0.227062192047, 3, 3, True),
            (0.175881617670, 3, 1, True),
            (-0.084490796177, 4, 3, False),
            (0.223541634759, 4, 1, False),
            (0.167466031764, 5, 5, True),
            (-0.274607982695, 5, 3, True),
            (0.300462606289, 5, 1, True),
            (-0.270030862434, 6, 5, False),
            (0.315328111943, 6, 3, False),
            (0.199431008804, 6, 1, False),
            (0.114108866147, 7, 7, True),
            (-0.251199047646, 7, 5, True),
            (-0.025246454204, 7, 3, True),
            (0.189488612701, 7, 1, True),
            (-0.114108866147, 8, 7, False),
            (0.135015431217, 8, 5, False),
            (-0.145029837779, 8, 3, False),
            (0.422475534112, 8, 1, False),
        ],
        (8, 5): [
            (0.449466574975, 2, 0, False),
            (-0.052148851565, 4, 4, False),
            (0.044073823779, 4, 0, False),
            (0.554700196225, 5, 4, True),
            (0.100503781526, 6, 4, False),
            (0.265908011739, 6, 0, False),
            (-0.438529009654, 7, 4, True),
            (-0.348608344389, 8, 8, False),
            (-0.032686022523, 8, 4, False),
            (0.299855896496, 8, 0, False),
        ],
        (8, 6): [
            (0.224733287488, 2, 1, False),
            (-0.257464325272, 3, 3, True),
            (0.332385014674, 3, 1, True),
            (-0.415148750267, 4, 3, False),
            (-0.156911478615, 4, 1, False),
            (0.063296210443, 5, 5, True),
            (-0.028306925854, 5, 3, True),
            (-0.061149948809, 5, 1, True),
            (-0.102062072616, 6, 5, False),
            (-0.198638039426, 6, 3, False),
            (0.175881617670, 6, 1, False),
            (-0.301903682212, 7, 7, True),
            (-0.094944315664, 7, 5, True),
            (0.448486349656, 7, 3, True),
            (0.115693787420, 7, 1, True),
            (0.301903682212, 8, 7, False),
            (0.051031036308, 8, 5, False),
            (0.237536546896, 8, 3, False),
            (0.146777248258, 8, 1, False),
        ],
        (8, 7): [
            (0.224733287488, 2, 1, True),
            (-0.257464325272, 3, 3, False),
            (-0.332385014674, 3, 1, False),
            (0.415148750267, 4, 3, True),
            (-0.156911478615, 4, 1, True),
            (-0.063296210443, 5, 5, False),
            (-0.028306925854, 5, 3, False),
            (0.061149948809, 5, 1, False),
            (-0.102062072616, 6, 5, True),
            (0.198638039426, 6, 3, True),
            (0.175881617670, 6, 1, True),
            (-0.301903682212, 7, 7, False),
            (0.094944315664, 7, 5, False),
            (0.448486349656, 7, 3, False),
            (-0.115693787420, 7, 1, False),
            (-0.301903682212, 8, 7, True),
            (0.051031036308, 8, 5, True),
            (-0.237536546896, 8, 3, True),
            (0.146777248258, 8, 1, True),
        ],
        (8, 8): [
            (0.449466574975, 2, 2, False),
            (-0.068278874200, 4, 2, False),
            (-0.554700196225, 5, 2, True),
            (0.235702260396, 6, 6, False),
            (0.158910431541, 6, 2, False),
            (0.322748612184, 7, 6, True),
            (-0.296885542998, 7, 2, True),
            (-0.220479275922, 8, 6, False),
            (-0.404843922270, 8, 2, False),
        ],
        (8, 9): [
            (0.224733287488, 2, 2, True),
            (0.420437482591, 3, 2, False),
            (0.443812682299, 4, 2, True),
            (0.092450032704, 5, 2, False),
            (0.235702260396, 6, 6, True),
            (-0.158910431541, 6, 2, True),
            (-0.322748612184, 7, 6, False),
            (0.458823111906, 7, 2, False),
            (-0.220479275922, 8, 6, True),
            (-0.350864732634, 8, 2, True),
        ],
        (12, 1): [
            (0.277350098113, 0, 0, False),
            (-0.238095850227, 4, 4, False),
            (-0.281718809194, 4, 0, False),
            (0.044386552658, 6, 4, False),
            (-0.016776539984, 6, 0, False),
            (0.359893076227, 8, 8, False),
            (0.236208667100, 8, 4, False),
            (0.444154851813, 8, 0, False),
            (0.310419274302, 10, 8, False),
            (0.260804356502, 10, 4, False),
            (-0.183014101628, 10, 0, False),
            (0.396143090090, 12, 8, False),
            (-0.190464572202, 12, 4, False),
            (0.105355538754, 12, 0, False),
        ],
        (12, 2): [
            (0.277350098113, 1, 0, False),
            (0.234669832041, 3, 0, False),
            (-0.127267585252, 4, 4, True),
            (-0.140699609418, 5, 4, False),
            (-0.214043172370, 5, 0, False),
            (0.023725610375, 6, 4, True),
            (-0.113599552374, 7, 4, False),
            (-0.316998329415, 7, 0, False),
            (0.384741882032, 8, 8, True),
            (0.126258843439, 8, 4, True),
            (0.380521195324, 9, 0, False),
            (0.331852163054, 10, 8, True),
            (0.139405792430, 10, 4, True),
            (0.172623178078, 11, 4, False),
            (-0.120034130258, 11, 0, False),
            (0.423494776930, 12, 8, True),
            (-0.101807596214, 12, 4, True),
        ],
        (12, 3): [
            (0.277350098113, 1, 1, False),
            (0.185522791844, 3, 3, False),
            (-0.143705336631, 3, 1, False),
            (0.044995886278, 4, 3, True),
            (0.119047925114, 4, 1, True),
            (-0.177972492663, 5, 5, False),
            (-0.184219031546, 5, 1, False),
            (-0.013910372102, 6, 5, True),
            (-0.016243814991, 6, 3, True),
            (0.010273490653, 6, 1, True),
            (-0.239044999132, 7, 7, False),
            (0.061811521145, 7, 5, False),
            (-0.124413732112, 7, 3, False),
            (0.192226030124, 7, 1, False),
            (-0.096185470508, 8, 7, True),
            (-0.113808183500, 8, 5, True),
            (-0.122249599488, 8, 3, True),
            (-0.356116131891, 8, 1, True),
            (0.231756202722, 9, 9, False),
            (-0.168627406449, 9, 7, False),
            (0.150824937476, 9, 5, False),
            (-0.142881128804, 9, 3, False),
            (0.139596066032, 9, 1, False),
            (-0.127854632553, 10, 9, True),
            (-0.152412838035, 10, 7, True),
            (-0.165314933662, 10, 5, True),
            (-0.172505867037, 10, 3, True),
            (0.181372686394, 10, 1, True),
            (0.146575492494, 11, 9, False),
            (-0.130235742729, 11, 7, False),
            (0.065245428532, 11, 5, False),
            (-0.038784847662, 11, 1, False),
            (-0.242587108962, 12, 9, True),
            (-0.264684235581, 12, 7, True),
            (0.148408799014, 12, 5, True),
            (0.152711394321, 12, 3, True),
            (-0.124339941908, 12, 1, True),
        ],
        (12, 4): [
            (0.277350098113, 1, 1, True),
            (-0.185522791844, 3, 3, True),
            (-0.143705336631, 3, 1, True),
            (0.044995886278, 4, 3, False),
            (-0.119047925114, 4, 1, False),
            (-0.177972492663, 5, 5, True),
            (-0.184219031546, 5, 1, True),
            (0.013910372102, 6, 5, False),
            (-0.016243814991, 6, 3, False),
            (-0.010273490653, 6, 1, False),
            (0.239044999132, 7, 7, True),
            (0.061811521145, 7, 5, True),
            (0.124413732112, 7, 3, True),
            (0.192226030124, 7, 1, True),
            (-0.096185470508, 8, 7, False),
            (0.113808183500, 8, 5, False),
            (-0.122249599488, 8, 3, False),
            (0.356116131891, 8, 1, False),
            (0.231756202722, 9, 9, True),
            (0.168627406449, 9, 7, True),
            (0.150824937476, 9, 5, True),
            (0.142881128804, 9, 3, True),
            (0.139596066032, 9, 1, True),
            (0.127854632553, 10, 9, False),
            (-0.152412838035, 10, 7, False),
            (0.165314933662, 10, 5, False),
            (-0.172505867037, 10, 3, False),
            (-0.181372686394, 10, 1, False),
            (0.146575492494, 11, 9, True),
            (0.130235742729, 11, 7, True),
            (0.065245428532, 11, 5, True),
            (-0.038784847662, 11, 1, True),
            (0.242587108962, 12, 9, False),
            (-0.264684235581, 12, 7, False),
            (-0.148408799014, 12, 5, False),
            (0.152711394321, 12, 3, False),
            (0.124339941908, 12, 1, False),
        ],
        (12, 5): [
            (0.118262479198, 2, 0, False),
            (-0.269975317671, 4, 4, False),
            (0.228170788401, 4, 0, False),
            (-0.397958591720, 5, 4, True),
            (0.050329619952, 6, 4, False),
            (0.133159657974, 6, 0, False),
            (-0.321308055294, 7, 4, True),
            (-0.136026796896, 8, 8, False),
            (0.267835453142, 8, 4, False),
            (-0.032218387227, 8, 0, False),
            (-0.117327457423, 10, 8, False),
            (0.295724343491, 10, 4, False),
            (0.222417248481, 10, 0, False),
            (0.488252079237, 11, 4, True),
            (-0.149728014282, 12, 8, False),
            (-0.215966524978, 12, 4, False),
            (0.172556152963, 12, 0, False),
        ],
        (12, 6): [
            (0.344791413584, 2, 1, False),
            (0.077935161166, 3, 3, True),
            (-0.100613860426, 3, 1, True),
            (0.081908896445, 4, 3, False),
            (0.030958652879, 4, 1, False),
            (-0.093454251368, 5, 5, True),
            (0.041794011769, 5, 3, True),
            (0.090285384340, 5, 1, True),
            (-0.105183451885, 6, 5, False),
            (0.095532740608, 6, 3, False),
            (-0.293469592827, 6, 1, False),
            (-0.234310966029, 7, 7, True),
            (0.090998078234, 7, 5, True),
            (0.000493724089, 7, 3, True),
            (0.261799658169, 7, 1, True),
            (0.094280619108, 8, 7, False),
            (0.015936333275, 8, 5, False),
            (0.074179594425, 8, 3, False),
            (0.045836638146, 8, 1, False),
            (0.292071235519, 9, 9, True),
            (-0.165287919211, 9, 7, True),
            (0.105598578061, 9, 5, True),
            (-0.060022079714, 9, 3, True),
            (-0.337890064569, 9, 1, True),
            (0.161129066053, 10, 9, False),
            (0.149394463156, 10, 7, False),
            (0.140435573581, 10, 5, False),
            (0.017254037856, 10, 3, False),
            (0.099635389136, 10, 1, False),
            (0.184722068652, 11, 9, True),
            (-0.127656561741, 11, 7, True),
            (-0.082225686732, 11, 5, True),
            (-0.132395933850, 11, 3, True),
            (0.131119011524, 11, 1, True),
            (0.305720907590, 12, 9, False),
            (0.259442444548, 12, 7, False),
            (-0.062344165666, 12, 5, False),
            (-0.085535489629, 12, 3, False),
            (-0.013541946779, 12, 1, False),
        ],
        (12, 7): [
            (0.344791413584, 2, 1, True),
            (0.077935161166, 3, 3, False),
            (0.100613860426, 3, 1, False),
            (-0.081908896445, 4, 3, True),
            (0.030958652879, 4, 1, True),
            (0.093454251368, 5, 5, False),
            (0.041794011769, 5, 3, False),
            (-0.090285384340, 5, 1, False),
            (-0.105183451885, 6, 5, True),
            (-0.095532740608, 6, 3, True),
            (-0.293469592827, 6, 1, True),
            (-0.234310966029, 7, 7, False),
            (-0.090998078234, 7, 5, False),
            (0.000493724089, 7, 3, False),
            (-0.261799658169, 7, 1, False),
            (-0.094280619108, 8, 7, True),
            (0.015936333275, 8, 5, True),
            (-0.074179594425, 8, 3, True),
            (0.045836638146, 8, 1, True),
            (-0.292071235519, 9, 9, False),
            (-0.165287919211, 9, 7, False),
            (-0.105598578061, 9, 5, False),
            (-0.060022079714, 9, 3, False),
            (0.337890064569, 9, 1, False),
            (0.161129066053, 10, 9, True),
            (-0.149394463156, 10, 7, True),
            (0.140435573581, 10, 5, True),
            (-0.017254037856, 10, 3, True),
            (0.099635389136, 10, 1, True),
            (-0.184722068652, 11, 9, False),
            (-0.127656561741, 11, 7, False),
            (0.082225686732, 11, 5, False),
            (-0.132395933850, 11, 3, False),
            (-0.131119011524, 11, 1, False),
            (0.305720907590, 12, 9, True),
            (-0.259442444548, 12, 7, True),
            (-0.062344165666, 12, 5, True),
            (0.085535489629, 12, 3, True),
            (-0.013541946779, 12, 1, True),
        ],
        (12, 8): [
            (0.118262479198, 2, 2, False),
            (-0.353480665427, 4, 2, False),
            (0.397958591720, 5, 2, True),
            (0.118033421305, 6, 6, False),
            (0.079578116410, 6, 2, False),
            (0.236476325732, 7, 6, True),
            (-0.217526581744, 7, 2, True),
            (0.229415733871, 8, 6, False),
            (-0.196584752617, 8, 2, False),
            (0.280115465323, 10, 10, False),
            (-0.140850450625, 10, 6, False),
            (0.228876666933, 10, 2, False),
            (-0.387802301437, 11, 10, True),
            (0.235387069246, 11, 6, True),
            (0.180533640939, 11, 2, True),
            (-0.268190139802, 12, 10, False),
            (-0.157879093190, 12, 6, False),
            (0.044540823344, 12, 2, False),
        ],
        (12, 9): [
            (0.344791413584, 2, 2, True),
            (-0.127267585252, 3, 2, False),
            (-0.087564293550, 4, 2, True),
            (-0.136498670851, 5, 2, False),
            (-0.313759753676, 6, 6, True),
            (-0.088708973422, 6, 2, True),
            (0.169378248534, 7, 6, False),
            (0.320986904326, 7, 2, False),
            (-0.068852829094, 8, 6, True),
            (-0.109570522537, 8, 2, True),
            (0.327185227372, 9, 6, False),
            (0.366740964879, 9, 2, False),
            (0.096078811959, 10, 10, True),
            (-0.006901615143, 10, 6, True),
            (0.262582509066, 10, 2, True),
            (0.133015092023, 11, 10, False),
            (0.265279038498, 11, 6, False),
            (-0.061922527983, 11, 2, False),
            (-0.091988459050, 12, 10, True),
            (0.320746948482, 12, 6, True),
            (-0.246620504243, 12, 2, True),
        ],
        (12, 10): [
            (0.154128699821, 3, 3, False),
            (0.198979295860, 3, 1, False),
            (0.261671903830, 4, 3, True),
            (-0.098902683233, 4, 1, True),
            (-0.207922583282, 5, 5, False),
            (-0.092985806055, 5, 3, False),
            (0.200872299226, 5, 1, False),
            (-0.281688823265, 6, 5, True),
            (0.183869963176, 6, 3, True),
            (-0.090684531264, 6, 1, True),
            (0.099296950387, 7, 7, False),
            (-0.065924608702, 7, 5, False),
            (-0.220739509137, 7, 3, False),
            (0.001812000214, 7, 1, False),
            (0.039954501988, 8, 7, True),
            (-0.236374021455, 8, 5, True),
            (-0.050781285780, 8, 3, True),
            (0.082181751016, 8, 1, True),
            (-0.131276257878, 9, 9, False),
            (0.070046172365, 9, 7, False),
            (0.113911274888, 9, 5, False),
            (-0.118702841805, 9, 3, False),
            (0.057986838015, 9, 1, False),
            (0.072422129448, 10, 9, True),
            (0.063310799522, 10, 7, True),
            (-0.066589288364, 10, 5, True),
            (-0.266155488449, 10, 3, True),
            (-0.209583396835, 10, 1, True),
            (-0.083026395519, 11, 9, False),
            (0.054098651431, 11, 7, False),
            (0.351098129009, 11, 5, False),
            (0.294562357294, 11, 3, False),
            (0.095200451808, 11, 1, False),
            (0.137411329233, 12, 9, True),
            (0.109947237986, 12, 7, True),
            (0.210162139587, 12, 5, True),
            (0.079293521097, 12, 3, True),
            (-0.153905354486, 12, 1, True),
        ],
        (12, 11): [
            (-0.154128699821, 3, 3, True),
            (0.198979295860, 3, 1, True),
            (0.261671903830, 4, 3, False),
            (0.098902683233, 4, 1, False),
            (-0.207922583282, 5, 5, True),
            (0.092985806055, 5, 3, True),
            (0.200872299226, 5, 1, True),
            (0.281688823265, 6, 5, False),
            (0.183869963176, 6, 3, False),
            (0.090684531264, 6, 1, False),
            (-0.099296950387, 7, 7, True),
            (-0.065924608702, 7, 5, True),
            (0.220739509137, 7, 3, True),
            (0.001812000214, 7, 1, True),
            (0.039954501988, 8, 7, False),
            (0.236374021455, 8, 5, False),
            (-0.050781285780, 8, 3, False),
            (-0.082181751016, 8, 1, False),
            (-0.131276257878, 9, 9, True),
            (-0.070046172365, 9, 7, True),
            (0.113911274888, 9, 5, True),
            (0.118702841805, 9, 3, True),
            (0.057986838015, 9, 1, True),
            (-0.072422129448, 10, 9, False),
            (0.063310799522, 10, 7, False),
            (0.066589288364, 10, 5, False),
            (-0.266155488449, 10, 3, False),
            (0.209583396835, 10, 1, False),
            (-0.083026395519, 11, 9, True),
            (-0.054098651431, 11, 7, True),
            (0.351098129009, 11, 5, True),
            (-0.294562357294, 11, 3, True),
            (0.095200451808, 11, 1, True),
            (-0.137411329233, 12, 9, False),
            (0.109947237986, 12, 7, False),
            (-0.210162139587, 12, 5, False),
            (0.079293521097, 12, 3, False),
            (0.153905354486, 12, 1, False),
        ],
        (12, 12): [
            (0.251691112854, 3, 2, False),
            (-0.279739031965, 4, 2, True),
            (-0.303690370876, 5, 2, False),
            (0.025020479683, 6, 6, True),
            (0.347497361598, 6, 2, True),
            (-0.194613659377, 7, 6, False),
            (0.158305641755, 7, 2, False),
            (0.218839820127, 8, 6, True),
            (-0.137516371971, 8, 2, True),
            (0.102941176471, 9, 6, False),
            (-0.204869843839, 9, 2, False),
            (0.323881609041, 10, 10, True),
            (-0.149828788952, 10, 6, True),
            (-0.033216518199, 10, 2, True),
            (0.448393783737, 11, 10, False),
            (-0.163298822463, 11, 6, False),
            (0.034591332447, 11, 2, False),
            (-0.310093032200, 12, 10, True),
            (-0.064593423593, 12, 6, True),
            (0.069157176743, 12, 2, True),
        ],
        (12, 13): [
            (0.366899692853, 3, 2, True),
            (0.291786426836, 6, 6, False),
            (-0.432789211451, 6, 2, False),
            (0.103162080537, 7, 6, True),
            (0.112149005260, 7, 2, True),
            (0.342997170285, 9, 6, True),
            (-0.164770510914, 9, 2, True),
            (-0.188853808195, 10, 10, False),
            (0.051550449553, 10, 6, False),
            (0.262856748205, 10, 2, False),
            (0.261456258292, 11, 10, True),
            (0.204040195343, 11, 6, True),
            (0.295595407165, 11, 2, True),
            (0.180813755368, 12, 10, False),
            (-0.286574784769, 12, 6, False),
            (0.072928584076, 12, 2, False),
        ],
    }
)
Hexagonal = Symmetry(
    [
        Quaternion(np.array(q, dtype=np.float64))
        for q in [
            [1.0, 0.0, 0.0, 0.0],
            [-0.5 * np.sqrt(3), 0.0, 0.0, -0.5],
            [0.5, 0.0, 0.0, 0.5 * np.sqrt(3)],
            [0.0, 0.0, 0.0, 1.0],
            [-0.5, 0.0, 0.0, 0.5 * np.sqrt(3)],
            [-0.5 * np.sqrt(3), 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -0.5 * np.sqrt(3), 0.5, 0.0],
            [0.0, 0.5, -0.5 * np.sqrt(3), 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -0.5, -0.5 * np.sqrt(3), 0.0],
            [0.0, 0.5 * np.sqrt(3), 0.5, 0.0],
        ]
    ],
    rgb_vectors=[
        [0, 0, 1],
        [1, 0, 0],
        [np.sqrt(3), 1, 0],
    ],
    fz_limiting_planes=[
        LimitingPlane(n, p)
        for n, p in [
            (np.array([0, 0, 1]), np.tan(np.pi / 12)),
            (np.array([0, 1, 0]), 1.0),
            (np.array([1, 0, 0]), 1.0),
            (np.array([np.sqrt(3) / 2.0, 0.5, 0]), 1.0),
            (np.array([0.5, np.sqrt(3) / 2.0, 0]), 1.0),
        ]
    ],
    expansion_table={},
)

symmetries = {"cubic": Cubic, "hexagonal": Hexagonal}



def test_project_to_standard_triangle():
    shape = [100, 100]
    ori_axis = _generate_axis_testdata(shape)
    oris_sst = Cubic.project_to_standard_triangle(ori_axis)
    assert tf.reduce_all(Cubic.in_standard_triangle(oris_sst))


def test_project_specific_to_fz():
    ori = Axis(tf.constant([[0, 0, -1, 1 * np.pi / 4]], dtype=tf.float64)).to_quaternion()
    ori = Cubic.project_to_fz(ori)
    assert Cubic.all_in_fz(ori, EPS=1e-12)


def test_project_to_fz():
    ori = _generate_axis_testdata([100, 100]).astype(Quaternion)
    assert not Cubic.all_in_fz(ori)
    ori = Cubic.project_to_fz(ori)
    assert Cubic.all_in_fz(ori, EPS=1e-12)


def test_project_to_fz_profile():
    ori_og = _generate_axis_testdata([100, 100, 100]).astype(Quaternion)
    assert not Cubic.all_in_fz(ori_og)

    ori = Cubic.project_all_to_fz(ori_og)
    assert Cubic.all_in_fz(ori, EPS=1e-12)
    ori2 = Cubic.project_all_to_fz_naive(ori_og)
    assert Cubic.all_in_fz(ori2, EPS=1e-12)



def test_project_specific_to_fz_SHSH():
    from mcrpy.src.SHSH import z_symm
    from mcrpy.orientation import Hypersphere

    ori = Axis(tf.constant([[0, 0, -1, 1 * np.pi / 4]], dtype=tf.float64)).to_quaternion()
    ori_hs = ori.astype(Hypersphere)
    for order, lambda_shsh in Cubic.expansion_table:
        s_1 = z_symm(ori_hs, Cubic, order, lambda_shsh)
        ori = Cubic.project_to_fz(ori)
        ori_hs = ori.astype(Hypersphere)
        s_1_fz = z_symm(ori_hs, Cubic, order, lambda_shsh)
        assert tf.reduce_max(tf.math.abs(s_1 - s_1_fz)) < 0.001


def test_project_to_fz_SHSH():
    from mcrpy.src.SHSH import z_symm
    from mcrpy.orientation import Hypersphere
    from mcrpy.orientation.test_orientation import _generate_axis_testdata

    ori = _generate_axis_testdata([100, 100]).to_quaternion()
    ori_og = ori
    for order, lambda_shsh in Cubic.expansion_table:
        s_1 = z_symm(ori_og, Cubic, order, lambda_shsh)
        ori = Cubic.project_to_fz(ori_og)
        s_1_fz = z_symm(ori, Cubic, order, lambda_shsh)
        assert tf.reduce_max(tf.math.abs(ori_og.x - ori.x)) > 0.001
        assert tf.reduce_max(tf.math.abs(s_1 - s_1_fz)) < 0.001


def test_appy_symmetries_SHSH():
    from mcrpy.src.SHSH import z_symm
    from mcrpy.orientation import Hypersphere
    from mcrpy.orientation.test_orientation import _generate_axis_testdata

    ori = _generate_axis_testdata([100, 100]).to_quaternion()
    for order, lambda_shsh in Cubic.expansion_table:
        s_1 = z_symm(ori, Cubic, order, lambda_shsh)
        oris_seq = Cubic.apply_to_orientation(ori)
        for ori_seq in oris_seq:
            s_1_fz = z_symm(ori_seq, Cubic, order, lambda_shsh)
            assert tf.reduce_max(tf.math.abs(s_1 - s_1_fz)) < 0.001


def plot_color_key(n_samples: int = 100, axis: bool = False, savefig: bool = True):  # sourcery skip: extract-method
    # mesh 2D grid
    phi_proj = np.arange(0, 0.25 * np.pi, 0.25 * np.pi / n_samples)
    # phi_proj = np.arange(0, 2.0*(1 + 1 / n_samples)*np.pi, 2*np.pi/n_samples)
    r_proj = np.arange(0, 0.7, 0.7 * 1 / n_samples)
    # r_proj = np.arange(0, 1*(1 + 1 / n_samples), 1 / n_samples)
    phi_proj_mg, r_proj_mg = np.meshgrid(phi_proj, r_proj)
    # convert to spherical coords
    phi = phi_proj_mg
    theta = 2 * np.arctan(r_proj_mg)
    # convert to vectors or unit length
    vec_x = np.sin(theta) * np.cos(phi)
    vec_y = np.sin(theta) * np.sin(phi)
    vec_z = np.cos(theta)
    vec = np.stack([vec_x, vec_y, vec_z], axis=-1).astype(np.float64)
    vs = tf.constant(vec, dtype=tf.float64)
    rgb = tf.einsum("ij,...j", Cubic.standard_triangle, vs)
    rgb = tf.math.sqrt(rgb)
    rgb_norm = tf.reduce_max(rgb, axis=-1, keepdims=True)
    rgb = tf.where(rgb_norm < 1e-4, ONE, rgb / rgb_norm).numpy()
    for i in range(n_samples):
        for j in range(n_samples):
            if np.isnan(rgb[i, j]).any():
                for c_index in range(3):
                    rgb[i, j, c_index] = 1
    # print(rgb.shape)
    f = rgb
    assert f.shape == (n_samples, n_samples, 3)
    # make plot
    if axis:
        import matplotlib

        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "font.size": 14,
                # 'font.size': 10,
                "figure.titlesize": "medium",
                "text.usetex": "True",
                "pgf.rcfonts": "False",
                "pgf.preamble": r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{mathrsfs}",
            }
        )
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=[2.5, 2.5] if axis else [3.5, 3.5])
    ax = fig.add_axes([0.01, 0.01, 0.99, 0.99], polar=True)
    if axis:
        ax.set_xticks([np.pi * i for i in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]])
        ax.set_yticks([])
    ax.pcolormesh(
        phi_proj_mg, r_proj_mg, f, shading="gouraud", zorder=100, edgecolors="white", antialiased=True
    )  # shading gouraud oder nearest
    # ax.pcolormesh(phi_proj_mg, r_proj_mg, f, edgecolors='face', cmap='seismic', vmin=-1, vmax=1, shading='gouraud') # shading gouraud oder nearest
    if axis:
        plt.text(0.5 * 45 / 180 * np.pi, 1.1, r"$\varphi$", zorder=1000)  # , transform=ax.transAxes)
        plt.text(0.62, 0.51, r"$\theta$", transform=ax.transAxes, zorder=1000)
        plt.text(0.5, 0.51, r"0°", transform=ax.transAxes, zorder=1000)
        plt.text(0.72, 0.51, r"60°", transform=ax.transAxes, zorder=1000)
        plt.text(0.92, 0.51, r"90°", transform=ax.transAxes, zorder=1000)
        plt.polar([i / 180 * np.pi for i in [-1, 1]], [0.5, 0.5], "k-", lw=0.5, zorder=1000)
        plt.polar([-0.5 * np.pi, 0.5 * np.pi], [0.01, 0.01], "k-", lw=0.5, zorder=1000)
        plt.polar([0 / 180 * np.pi] * 2, [0, 1], "k-", lw=0.5, zorder=1000)
    else:
        # pass
        ax.set_xticks([])
        ax.set_yticks([])
        # plt.axis('off') # wenn nichtmal schwarzer kreis erwünscht
        # plt.title(r'$\omega = $' + f'{omega_deg:.1f}°')
    if savefig:
        plt.savefig(f"ipf_key.png", bbox_inches="tight", dpi=600)  # dpi 600
    else:
        plt.show()
    plt.close()

