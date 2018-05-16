"""
Class for the space of matrices (m, n),
i.e. the (m*n)-dimensional Euclidean space.
"""

import geomstats.backend as gs

from geomstats.euclidean_space import EuclideanSpace


class MatrixSpace(EuclideanSpace):
    """
    Class for the space of matrices (m, n),
    i.e. the (m*n)-dimensional Euclidean space.
    """

    def __init__(self, m, n):
        assert isinstance(m, int) and isinstance(n, int) and m > 0 and n > 0
        super().init(dimension=m*n)
        self.m = m
        self.n = n

    def belongs(self, point):
        """
        Evaluate if a point belongs to the matrix space.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        _, mat_dim_1, mat_dim_2 = point.shape
        return mat_dim_1 == self.m & mat_dim_2 == self.n

    def vector_from_matrix(self, matrix):
        """
        Convert a matrix to a vector.
        """
        matrix = gs.to_ndarray(matrix, to_ndim=3)
        n_mats, m, n = matrix.shape
        return gs.reshape(matrix, (n_mats, m*n))
