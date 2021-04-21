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

from sknetwork.utils.base import Algorithm
from sknetwork.utils.format import directed2undirected
from sknetwork.utils.knn1d import knn1d


class BaseTransformer(Algorithm, ABC):
    """Base class for transformers."""
    def __init__(self, undirected: bool = False):
        self.undirected = undirected

        self.adjacency_ = None

    def fit_transform(self, x: np.ndarray) -> sparse.csr_matrix:
        """Fit algorithm to the data and return the computed adjacency.

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
            dtype = self.adjacency_.dtype
            self.adjacency_ = directed2undirected(self.adjacency_, weighted=False).astype(dtype)

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
        Return approximate nearest neighbors; the k-th returned value is guaranteed to be no further than (1+tol_nn)
        times the distance to the real k-th nearest neighbor.
    n_jobs :
        Number of jobs to schedule for parallel processing. If -1 is given all processors are used.

    Attributes
    ----------
    adjacency_ :
        Adjacency matrix of the graph.

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
        """Fit algorithm to the data.

        Parameters
        ----------
        x :
            Data to transform into adjacency.

        Returns
        -------
        self : :class:`KNNDense`
        """
        tree = cKDTree(x, self.leaf_size)
        _, neighbors = tree.query(x, self.n_neighbors + 1, self.eps, self.p, workers=self.n_jobs)

        n: int = x.shape[0]
        indptr: np.ndarray = np.arange(n + 1) * (self.n_neighbors + 1)
        indices: np.ndarray = neighbors.reshape(-1)
        data = np.ones(indices.shape[0], dtype=bool)

        self.adjacency_ = sparse.csr_matrix((data, indices, indptr))
        self.make_undirected()
        self.adjacency_.setdiag(0)

        return self


class CNNDense(BaseTransformer):
    """Extract adjacency from vector data through component-wise k-nearest-neighbor search.
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
    adjacency_ :
        Adjacency matrix of the  graph.
    """
    def __init__(self, n_neighbors: int = 1, undirected: bool = False):
        super(CNNDense, self).__init__(undirected)

        self.n_neighbors = n_neighbors

    def fit(self, x: np.ndarray) -> 'CNNDense':
        """Fit algorithm to the data.

        Parameters
        ----------
        x:
            Data to transform into adjacency.

        Returns
        -------
        self: :class:`CNNDense`
        """
        rows, cols = [], []
        for j in range(x.shape[1]):
            row, col = knn1d(x[:, j].astype(np.float32), self.n_neighbors)
            rows += row
            cols += col

        rows = np.array(rows)
        cols = np.array(cols)
        data = np.ones(cols.shape[0], dtype=bool)

        self.adjacency_ = sparse.csr_matrix((data, (rows, cols)))
        self.make_undirected()

        return self
