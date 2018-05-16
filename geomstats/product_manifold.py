"""
Class for a product of manifolds.
"""

import geomstats.backend as gs

from geomstats.manifold import Manifold

# TODO(nina): get rid of for loops
# TODO(nina): unit tests


class ProductManifold(Manifold):
    """
    Class for a product of manifolds.
    """

    def __init__(self, manifolds):
        self.manifolds = manifolds
        self.n_manifolds = len(manifolds)
        dimensions = [manifold.dimension for manifold in manifolds]
        super(ProductManifold, self).__init__(
            dimension=gs.sum(dimensions))

    def belongs(self, point):
        """
        Evaluate if a point belongs to the product of manifolds.
        """
        belong = [self.manifold[i].belongs(point[i])
                  for i in range(self.n_manifolds)]
        return gs.all(belong)

    def regularize(self, point):
        """
        Regularize a point to the canonical representation
        chosen for the product of manifolds.
        """
        regularize_points = [self.manifold[i].regularize(point[i])
                             for i in range(self.n_manifolds)]
        return regularize_points
