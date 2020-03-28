#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from sknetwork import njit, prange
from sknetwork.utils.formats import directed2undirected
from sknetwork.utils.base import Algorithm


class BaseTransformer(Algorithm, ABC):
    """Base class for transformers."""

    def __init__(self, undirected: bool = False):
        self.undirected = undirected

        self.adjacency_ = None

    def fit_transform(self, x: np.ndarray) -> sparse.csr_matrix:
        """Fits the model to the data and return the computed adjacency.

        Parameters
        ----------
        x: np.ndarray
            Input data.

        Returns
        -------
        adjacency : sparse.csr_matrix

        """
        self.fit(x)
        return self.adjacency_

    def make_undirected(self):
        """Modifies the adjacency to match desired constrains."""
        if self.adjacency_ is not None and self.undirected:
            self.adjacency_ = directed2undirected(self.adjacency_, weight_sum=False).astype(int)

        return self


class KNNDense(BaseTransformer):
    """Extract adjacency from vector data through k-nearest-neighbor search with KD-Tree.

    Parameters
    ----------
    n_neighbors :
        Number of neighbors for each sample in the transformed sparse graph.
    undirected :
        As the nearest neighbor relationship is not symmetric, the graph is directed by default.
        Setting this parameter to ``True`` forces the algorithm to return undirected graphs.
    leaf_size :
        Leaf size passed to KDTree.
        This can affect the speed of the construction and query, as well as the memory required to store the tree.
    p :
        Which Minkowski p-norm to use. 1 is the sum-of-absolute-values “Manhattan” distance,
        2 is the usual Euclidean distance infinity is the maximum-coordinate-difference distance.
        A finite large p may cause a ValueError if overflow can occur.
    eps :
        Return approximate nearest neighbors; the k-th returned value is guaranteed to be no further than (1+eps) times
        the distance to the real k-th nearest neighbor.
    n_jobs :
        Number of jobs to schedule for parallel processing. If -1 is given all processors are used.

    Attributes
    ----------
    adjacency_ :
        Adjacency matrix of the computed graph.

    References
    ----------
    Maneewongvatana, S., & Mount, D. M. (1999, December). It’s okay to be skinny, if your friends are fat.
    In Center for Geometric Computing 4th Annual Workshop on Computational Geometry (Vol. 2, pp. 1-8).


    """

    def __init__(self, n_neighbors: int = 5, undirected: bool = False, leaf_size: int = 16, p=2, eps: float = 0.01,
                 n_jobs=1):
        super(KNNDense, self).__init__(undirected)

        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.p = p
        self.eps = eps
        self.n_jobs = n_jobs

    def fit(self, x: np.ndarray) -> 'KNNDense':
        """

        Parameters
        ----------
        x :
            Data to transform into adjacency.

        Returns
        -------

        self : :class:`KNNDense`

        """
        tree = cKDTree(x, self.leaf_size)
        _, neighbors = tree.query(x, self.n_neighbors + 1, self.eps, self.p, n_jobs=self.n_jobs)

        n: int = x.shape[0]
        indptr: np.ndarray = np.arange(n + 1) * (self.n_neighbors + 1)
        indices: np.ndarray = neighbors.reshape(-1)
        data = np.ones(len(indices))

        self.adjacency_ = sparse.csr_matrix((data, indices, indptr))
        self.make_undirected()
        self.adjacency_.setdiag(0)

        return self


@njit
def knn1d(x: np.ndarray, n_neighbors: int) -> list:
    """K nearest neighbors search for 1-dimensional arrays.

    Parameters
    ----------
    x : np.ndarray
        1-d data
    n_neighbors : int
        Number of neighbors to return.
    Returns
    -------
    list :
        List of nearest neighbors tuples (i, j).

    """
    sorted_ix = np.argsort(x)
    edgelist = []

    n: int = x.shape[0]
    for i in prange(n):
        ix = sorted_ix[i]
        low: int = max(0, i - n_neighbors)
        hgh: int = min(n - 1, i + n_neighbors + 1)
        candidates = sorted_ix[low:hgh]

        deltas = np.abs(x[candidates] - x[ix])
        sorted_candidates = candidates[np.argsort(deltas)]
        sorted_candidates = sorted_candidates[:n_neighbors+1]
        edgelist.extend([(ix, neigh) for neigh in sorted_candidates if neigh != ix])

    return edgelist


class PNNDense(BaseTransformer):
    """Extract adjacency from vector data through parallel k-nearest-neighbor search.
    KNN is applied independently on each column of the input matrix.

    Parameters
    ----------
    n_neighbors :
        Number of neighbors per dimension.
    undirected :
        As the nearest neighbor relationship is not symmetric, the graph is directed by default.
        Setting this parameter to ``True`` forces the algorithm to return undirected graphs.

    Attributes
    ----------
    adjacency_:
        Adjacency matrix of the computed graph.
    """

    def __init__(self, n_neighbors: int = 1, undirected: bool = False):
        super(PNNDense, self).__init__(undirected)

        self.n_neighbors = n_neighbors

    def fit(self, x: np.ndarray) -> 'PNNDense':
        """

        Parameters
        ----------
        x:
            Data to transform into adjacency.

        Returns
        -------

        self: :class:`PNNDense`

        """

        edgelist = []
        for j in range(x.shape[1]):
            edgelist += knn1d(x[:, j], self.n_neighbors)
        edges = np.array(edgelist)
        data = np.ones(edges.shape[0])
        row = edges[:, 0]
        col = edges[:, 1]

        self.adjacency_ = sparse.csr_matrix((data, (row, col)))
        self.make_undirected()

        return self
