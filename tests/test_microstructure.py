import unittest
import os
from copy import deepcopy

import numpy as np
import tensorflow as tf

import mcrpy
from mcrpy.src.Microstructure import Microstructure

from example_data import example_ms

class TestMicrostructureFromFile(unittest.TestCase):
    def test_from_file_creation(self):
        fname_existing_ms = example_ms
        fname_wrong_ending = fname_existing_ms[:-3] + 'png'
        fname_readme = os.path.join(
            os.path.dirname(fname_existing_ms),
            'README.md'
        ) 
        self.assertTrue(os.path.isfile(fname_existing_ms))
        self.assertTrue(os.path.isfile(fname_readme))
        self.assertFalse(os.path.isfile(fname_wrong_ending))
        for creation_function in [
            Microstructure.from_npy,
            Microstructure.from_pickle,
            Microstructure.load,
            mcrpy.load
        ]:
            for fname in [
                fname_readme,
                fname_wrong_ending
            ]:
                with self.assertRaises(Exception):
                    creation_function(fname)
        for creation_function in [
            Microstructure.from_npy,
            Microstructure.load,
            mcrpy.load
        ]:
            ms = creation_function(fname_existing_ms)
            self.assertIsInstance(ms, Microstructure)
        with self.assertRaises(Exception):
            Microstructure.from_pickle(fname_existing_ms)

class TestMicrostructureFromArray(unittest.TestCase):
    def setUp(self):
        self.ms_npy = np.load(example_ms)
    
    def test_example_base_ms(self):
        self.assertEqual(len(self.ms_npy.shape), 2)
        self.assertEqual(self.ms_npy.shape[0], 64)
        self.assertEqual(self.ms_npy.shape[1], 64)
        self.assertEqual(np.min(self.ms_npy), 0)
        self.assertEqual(np.max(self.ms_npy), 1)
        self.assertEqual(np.unique(self.ms_npy).size, 2)
        
    def test_multiphase_default(self):
        ms = Microstructure(self.ms_npy)
        self.assertEqual(ms.n_phases, 1)
        
    def test_multiphase_enabled(self):
        ms = Microstructure(self.ms_npy, use_multiphase=True)
        self.assertEqual(ms.n_phases, 2)
        
    def test_multiphase_disabled(self):
        ms = Microstructure(self.ms_npy, use_multiphase=False)
        self.assertEqual(ms.n_phases, 1)
        
    def test_trainable_default(self):
        ms = Microstructure(self.ms_npy)
        self.assertIsInstance(ms.x, tf.Variable)
        self.assertTrue(ms.x.trainable)
        
    def test_trainable_enabled(self):
        ms = Microstructure(self.ms_npy, trainable=True)
        self.assertIsInstance(ms.x, tf.Variable)
        self.assertTrue(ms.x.trainable)
        
    def test_trainable_disabled(self):
        ms = Microstructure(self.ms_npy, trainable=False)
        self.assertIsInstance(ms.x, tf.Variable)
        self.assertFalse(ms.x.trainable)

class TestMicrostructureMultiphaseSinglephase(unittest.TestCase):
    rng = np.random.default_rng(42)
    ms_binary_2d = rng.integers(0, high=2, size=(5, 5))
    ms_binary_3d = rng.integers(0, high=2, size=(5, 4, 6))
    ms_threephase_2d = rng.integers(0, high=3, size=(5, 5))
    ms_threephase_3d = rng.integers(0, high=3, size=(5, 4, 6))
    ms_cases = [
        ms_binary_2d,
        ms_binary_3d,
        ms_threephase_2d,
        ms_threephase_3d
    ]
    ms_binary_cases = [
        ms_binary_2d,
        ms_binary_3d
    ]
        
    def test_general_flags(self):
        for ms_case in self.ms_cases:
            ms = Microstructure(ms_case)
            self.assertTrue(ms.has_phases)
            self.assertFalse(ms.has_orientations)
        
    def test_ms_binary_2d(self):
        ms = Microstructure(self.ms_binary_2d)
        self.assertTrue(ms.is_2D)
        self.assertFalse(ms.is_3D)
        self.assertEqual(ms.spatial_shape, (5, 5))
        self.assertEqual(ms.shape, (5, 5, 1))
        self.assertEqual(ms.extra_shape, 1)
        self.assertEqual(ms.x_shape, (1, 5, 5, 1))
        self.assertEqual(ms.n_phases, 1)
        
    def test_ms_binary_3d(self):
        ms = Microstructure(self.ms_binary_3d)
        self.assertFalse(ms.is_2D)
        self.assertTrue(ms.is_3D)
        self.assertEqual(ms.spatial_shape, (5, 4, 6))
        self.assertEqual(ms.shape, (5, 4, 6, 1))
        self.assertEqual(ms.extra_shape, 1)
        self.assertEqual(ms.x_shape, (1, 5, 4, 6, 1))
        self.assertEqual(ms.n_phases, 1)
        
    def test_ms_threephase_2d(self):
        ms = Microstructure(self.ms_threephase_2d)
        self.assertTrue(ms.is_2D)
        self.assertFalse(ms.is_3D)
        self.assertEqual(ms.spatial_shape, (5, 5))
        self.assertEqual(ms.shape, (5, 5, 3))
        self.assertEqual(ms.extra_shape, 3)
        self.assertEqual(ms.x_shape, (1, 5, 5, 3))
        self.assertEqual(ms.n_phases, 3)
        
    def test_ms_threephase_3d(self):
        ms = Microstructure(self.ms_threephase_3d)
        self.assertFalse(ms.is_2D)
        self.assertTrue(ms.is_3D)
        self.assertEqual(ms.spatial_shape, (5, 4, 6))
        self.assertEqual(ms.shape, (5, 4, 6, 3))
        self.assertEqual(ms.extra_shape, 3)
        self.assertEqual(ms.x_shape, (1, 5, 4, 6, 3))
        self.assertEqual(ms.n_phases, 3)
        
    def test_ms_binary_3d_blockshapes(self):
        ms = Microstructure(self.ms_binary_3d)
        self.assertIsInstance(ms.paddings, tf.Tensor)
        self.assertIsInstance(ms.block_shapes, tf.Tensor)
        self.assertIsInstance(ms.batch_element_shapes, list)
        self.assertIsInstance(ms.swapped_index_1, tf.Variable)
        self.assertIsInstance(ms.swapped_index_2, tf.Variable)
        self.assertEqual(ms.batch_element_shapes[0][0], 1)
        self.assertEqual(ms.batch_element_shapes[0][1], 4)
        self.assertEqual(ms.batch_element_shapes[0][2], 6)
        self.assertEqual(ms.batch_element_shapes[0][3], 1)
        self.assertEqual(ms.batch_element_shapes[1][0], 1)
        self.assertEqual(ms.batch_element_shapes[1][1], 5)
        self.assertEqual(ms.batch_element_shapes[1][2], 6)
        self.assertEqual(ms.batch_element_shapes[1][3], 1)
        self.assertEqual(ms.batch_element_shapes[2][0], 1)
        self.assertEqual(ms.batch_element_shapes[2][1], 5)
        self.assertEqual(ms.batch_element_shapes[2][2], 4)
        self.assertEqual(ms.batch_element_shapes[2][3], 1)
    
    def test_iterator_completeness(self):
        for ms_test_case in self.ms_binary_cases:
            ms = Microstructure(ms_test_case)
            cpy_ms_test_case = deepcopy(ms_test_case)
            for idx in ms:
                cpy_ms_test_case[idx] = cpy_ms_test_case[idx] + 1
            differences = np.unique(cpy_ms_test_case - ms_test_case)
            self.assertIn(1, differences)
            self.assertEqual(differences.size, 1)
    
    def test_encoding_context_manager_shapes(self):
        for ms_test_case in self.ms_binary_cases:
            ms = Microstructure(ms_test_case)
            self.assertEqual(ms.x.numpy().shape[-1], 1)
            with ms.use_multiphase_encoding() as x:
                self.assertEqual(x.numpy().shape[-1], 2)
            self.assertEqual(ms.x.numpy().shape[-1], 1)
            with ms.use_multiphase_encoding() as x:
                self.assertEqual(x.numpy().shape[-1], 2)
                with ms.use_singlephase_encoding() as xx:
                    self.assertEqual(xx.numpy().shape[-1], 1)
            self.assertEqual(ms.x.numpy().shape[-1], 1)
            with ms.use_singlephase_encoding() as xx:
                self.assertEqual(xx.numpy().shape[-1], 1)
            self.assertEqual(ms.x.numpy().shape[-1], 1)