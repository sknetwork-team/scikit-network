#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from sknetwork.utils.adjacency_formats import directed2undirected
from sknetwork.utils.algorithm_base_class import Algorithm


class KNeighborsTransformer(Algorithm):
    """Extract adjacency from vector data through KNN search with KD-Tree.

    Parameters
    ----------
    n_neighbors:
        Number of neighbors for each sample in the transformed sparse graph. As each sample is its own neighbor,
        one extra neighbor will be computed such that the sparse graph contains (n_neighbors + 1) neighbors.
    make_undirected:
        As the nearest neighbor relationship is not symmetric, the graph is directed by default.
        Setting this parameter to ``True`` forces the algorithm to return undirected graphs.
    remove_self_loops:
        If ``True`` the diagonal of the adjacency is set to 0.
    leaf_size:
        Leaf size passed to KDTree.
        This can affect the speed of the construction and query, as well as the memory required to store the tree.
    p:
        Which Minkowski p-norm to use. 1 is the sum-of-absolute-values “Manhattan” distance,
        2 is the usual Euclidean distance infinity is the maximum-coordinate-difference distance.
        A finite large p may cause a ValueError if overflow can occur.
    eps:
        Return approximate nearest neighbors; the k-th returned value is guaranteed to be no further than (1+eps) times
        the distance to the real k-th nearest neighbor.
    n_jobs:
        Number of jobs to schedule for parallel processing. If -1 is given all processors are used.

    Attributes
    ----------
    adjacency_:
        Adjacency matrix of the computed graph.

    References
    ----------
    Maneewongvatana, S., & Mount, D. M. (1999, December). It’s okay to be skinny, if your friends are fat.
    In Center for Geometric Computing 4th Annual Workshop on Computational Geometry (Vol. 2, pp. 1-8).


    """

    def __init__(self, n_neighbors: int = 5, make_undirected: bool = False, remove_self_loops: bool = True,
                 leaf_size: int = 16, p=2, eps: float = 0.01, n_jobs=1):
        self.n_neighbors = n_neighbors
        self.make_undirected = make_undirected
        self.remove_self_loops = remove_self_loops
        self.leaf_size = leaf_size
        self.p = p
        self.eps = eps
        self.n_jobs = n_jobs

        self.adjacency_ = None

    def fit(self, x: np.ndarray) -> 'KNeighborsTransformer':
        """

        Parameters
        ----------
        x:
            Data to transform into adjacency.

        Returns
        -------

        self: :class:`KNeighborsTransformer`

        """
        tree = cKDTree(x, self.leaf_size)
        _, neighbors = tree.query(x, self.n_neighbors + 1, self.eps, self.p, n_jobs=self.n_jobs)

        n: int = x.shape[0]
        indptr: np.ndarray = np.arange(n + 1) * (self.n_neighbors + 1)
        indices: np.ndarray = neighbors.reshape(-1)
        data = np.ones(len(indices))

        adjacency = sparse.csr_matrix((data, indices, indptr))
        if self.make_undirected:
            self.adjacency_ = directed2undirected(adjacency, weight_sum=False).astype(int)
        else:
            self.adjacency_ = adjacency

        if self.remove_self_loops:
            self.adjacency_.setdiag(0)

        return self
