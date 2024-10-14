import unittest

import numpy as np

import mcrpy

from example_data import example_ms

class TestLinealPath(unittest.TestCase):
    def setUp(self):
        self.ms = mcrpy.Microstructure.from_npy(example_ms)
        self.lineal_path_types = [
            'LinealPath', 
            'LinealPathApproximation', 
            'LineLinealPathApproximation'
        ]
    
    def test_vf_consistency(self):
        descriptors = mcrpy.characterize(
            self.ms,
            settings=mcrpy.CharacterizationSettings(
                descriptor_types=self.lineal_path_types + ['VolumeFractions'],
                limit_to=4,
                use_multigrid_descriptor=False,
                use_multiphase=False,
                periodic=True
            )
        )
        for lineal_path_type in self.lineal_path_types:
            lp = descriptors[lineal_path_type]
            vf = descriptors['VolumeFractions']
            self.assertEqual(vf.size, 1)
            vf = vf.flatten()[0]
            max_lp = np.max(lp)
            self.assertAlmostEqual(vf, max_lp)
    
    # TODO: implement index sorting s.t. 0 leq lp leq s2 forall r can be validated easily
    def test_range(self):
        descriptors = mcrpy.characterize(
            self.ms,
            settings=mcrpy.CharacterizationSettings(
                descriptor_types=self.lineal_path_types + ['FFTCorrelations'],
                limit_to=16,
                use_multigrid_descriptor=False,
                use_multiphase=False,
                periodic=True
            )
        )
        s2 = descriptors['FFTCorrelations'] # tested separately
        max_s2= np.max(s2)
        for lineal_path_type in self.lineal_path_types:
            lp = descriptors[lineal_path_type]
            self.assertTrue((lp > 0).all())
            max_lp= np.max(lp)
            self.assertAlmostEqual(max_lp, max_s2)
