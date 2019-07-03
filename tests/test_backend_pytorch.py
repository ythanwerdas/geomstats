"""
Unit tests for numpy backend.
"""

import importlib
import os
import unittest
import warnings

import geomstats.backend as gs
from geomstats.spd_matrices_space import SPDMatricesSpace


class TestBackendPytorch(unittest.TestCase):
    _multiprocess_can_split_ = True

    @classmethod
    def setUpClass(cls):
        cls.initial_backend = os.environ['GEOMSTATS_BACKEND']
        os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
        importlib.reload(gs)

    @classmethod
    def tearDownClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = cls.initial_backend
        importlib.reload(gs)

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

        self.spd = SPDMatricesSpace(n=3)
        self.n_samples = 2

    def test_sqrtm(self):
        point = gs.array([[4., 0., 0.],
                          [0., 9., 0.],
                          [0., 0., 16.]])
        result = gs.linalg.sqrtm(point)
        expected = gs.array(
            [[2., 0., 0.],
             [0., 3., 0.],
             [0., 0., 4.]])

        self.assertTrue(gs.allclose(result, expected))

    def test_logm(self):
        point = gs.array([[2., 0., 0.],
                          [0., 3., 0.],
                          [0., 0., 4.]])
        result = gs.linalg.logm(point)
        expected = gs.array([[0.693147180, 0., 0.],
                             [0., 1.098612288, 0.],
                             [0., 0., 1.38629436]])

        self.assertTrue(gs.allclose(result, expected))

    def test_expm_and_logm(self):
        point = gs.array([[2., 0., 0.],
                          [0., 3., 0.],
                          [0., 0., 4.]])
        result = gs.linalg.expm(gs.linalg.logm(point))
        expected = point

        self.assertTrue(gs.allclose(result, expected))

    def test_expm_vectorization(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])

        expected = gs.array([[[7.38905609, 0., 0.],
                              [0., 20.0855369, 0.],
                              [0., 0., 54.5981500]],
                             [[2.718281828, 0., 0.],
                              [0., 148.413159, 0.],
                              [0., 0., 403.42879349]]])

        result = gs.linalg.expm(point)

        self.assertTrue(gs.allclose(result, expected))

    def test_logm_vectorization_diagonal(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])

        expected = gs.array([[[0.693147180, 0., 0.],
                              [0., 1.09861228866, 0.],
                              [0., 0., 1.38629436]],
                             [[0., 0., 0.],
                              [0., 1.609437912, 0.],
                              [0., 0., 1.79175946]]])

        result = gs.linalg.logm(point)

        self.assertTrue(gs.allclose(result, expected))

    def test_expm_and_logm_vectorization_random_rotation(self):
        point = self.spd.random_uniform(self.n_samples)

        result = gs.linalg.expm(gs.linalg.logm(point))
        expected = point

        self.assertTrue(gs.allclose(result, expected))

    def test_expm_and_logm_vectorization(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])
        result = gs.linalg.expm(gs.linalg.logm(point))
        expected = point

        self.assertTrue(gs.allclose(result, expected))


if __name__ == '__main__':
        unittest.main()
