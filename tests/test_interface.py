import unittest
import mcrpy

class TestRequiresInput(unittest.TestCase):
    def test_characterize(self):
        with self.assertRaises(Exception):
            mcrpy.characterize(None)
    
    def test_reconstruct(self):
        with self.assertRaises(Exception):
            mcrpy.reconstruct(None)
    
    def test_match(self):
        with self.assertRaises(Exception):
            mcrpy.match(None)
    
    def test_view(self):
        with self.assertRaises(Exception):
            mcrpy.view(None)
    
    def test_smooth(self):
        with self.assertRaises(Exception):
            mcrpy.smooth(None)
    
    def test_merge(self):
        with self.assertRaises(Exception):
            mcrpy.merge(None)
    
    def test_interpolate(self):
        with self.assertRaises(Exception):
            mcrpy.interpolate(None)
            