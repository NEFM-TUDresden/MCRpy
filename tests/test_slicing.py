import unittest

import numpy as np

import mcrpy

from example_data import example_ms

def sliced_vf(slice_mode, isotropic):
    ms = mcrpy.Microstructure(
        np.array(
            list(range(8))
        ).reshape((2,)*3) / 10,
        use_multiphase=False
    )
    vf = mcrpy.characterize(
        ms,
        settings=mcrpy.CharacterizationSettings(
            descriptor_types=['VolumeFractions'],
            use_multigrid_descriptor=False,
            use_multiphase=False,
            periodic=True,
            slice_mode=slice_mode,
            isotropic=isotropic
        )
    )['VolumeFractions']
    return vf

class TestPermutationLoop(unittest.TestCase):
    
    def test_slice_mode_average_isotropic(self):
        vf = sliced_vf('average', True)
        self.assertEqual(vf.size, 1)
        self.assertAlmostEqual(vf[0][0], 0.35)
    
    def test_slice_mode_average_anisotropic(self):
        vf = sliced_vf('average', False)
        self.assertIsInstance(vf, tuple)
        self.assertEqual(len(vf), 3)
        for dir_vf in vf:
            self.assertEqual(dir_vf.size, 1)
            self.assertAlmostEqual(dir_vf[0][0], 0.35)
    
    def test_slice_mode_samplesurface_isotropic(self):
        vf = sliced_vf('sample_surface', True)
        self.assertEqual(vf.size, 1)
        self.assertAlmostEqual(vf[0][0], 0.25)
    
    def test_slice_mode_samplesurface_anisotropic(self):
        vf = sliced_vf('sample_surface', False)
        self.assertIsInstance(vf, tuple)
        self.assertEqual(len(vf), 3)
        for dir_vf, dir_vf_expected in zip(vf, [0.15, 0.25, 0.3]):
            self.assertEqual(dir_vf.size, 1)
            self.assertAlmostEqual(dir_vf[0][0], dir_vf_expected)
            