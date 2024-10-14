import unittest

import numpy as np

import mcrpy

from example_data import example_ms

class TestFFTCorrelations(unittest.TestCase):
    def setUp(self):
        self.ms = mcrpy.Microstructure.from_npy(example_ms)
    
    def test_vf_consistency(self):
        descriptors = mcrpy.characterize(
            self.ms,
            settings=mcrpy.CharacterizationSettings(
                descriptor_types=['FFTCorrelations', 'VolumeFractions'],
                limit_to=8,
                use_multigrid_descriptor=False,
                use_multiphase=False,
                periodic=True
            )
        )
        s2 = descriptors['FFTCorrelations']
        vf = descriptors['VolumeFractions']
        self.assertEqual(vf.size, 1)
        vf = vf.flatten()[0]
        vf_np = np.average(self.ms.numpy())
        max_s2 = np.max(s2)
        self.assertAlmostEqual(vf, max_s2)
        self.assertAlmostEqual(vf_np, max_s2)
    
    def test_range(self):
        descriptors = mcrpy.characterize(
            self.ms,
            settings=mcrpy.CharacterizationSettings(
                descriptor_types=['FFTCorrelations'],
                limit_to=32,
                use_multigrid_descriptor=False,
                use_multiphase=False,
                periodic=True
            )
        )
        s2 = descriptors['FFTCorrelations']
        self.assertTrue((s2 > 0).all())
        max_s2 = np.max(s2)
        ind = np.unravel_index(np.argmax(s2, axis=None), s2.shape)
        self.assertEqual(ind, (0, 0, 31, 31))
        s2[ind] = 0
        self.assertTrue((s2 < max_s2).all())
