#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.linkpred.base import BaseLinkPred
from sknetwork.utils.check import check_format


class FirstOrder(BaseLinkPred, ABC):
    """Base class for first order algorithms."""
    def __init__(self):
        super(FirstOrder, self).__init__()
        self.indptr_ = None
        self.indices_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]):
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph

        Returns
        -------
        self : :class:`FirstOrder`
        """
        adjacency = check_format(adjacency)
        self.indptr_ = adjacency.indptr
        self.indices_ = adjacency.indices

        return self

    def _neighborhood(self, node: int):
        """Out neighbors of a given node.

        Parameters
        ----------
        node : int
            Node to query.

        Returns
        -------
        neighbors : np.ndarray
            Out-neighbors.
        """
        if self.indptr_ is None or self.indices_ is None:
            raise ValueError("Please call the fit method first.")
        return self.indices_[self.indptr_[node]:self.indptr_[node+1]]


class CommonNeighbors(FirstOrder):
    """Link prediction by common neighbors:

    :math:`s(i, j) = |\\Gamma_i \\cap \\Gamma_j|`.

    Attributes
    ----------
    indptr_ : np.ndarray
        Pointer index for neighbors.
    indices_ : np.ndarray
        Concatenation of neighbors.

    Examples
    --------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> cn = CommonNeighbors()
    >>> similarities = cn.fit_predict(adjacency, 0)
    >>> similarities
    array([2, 1, 1, 1, 1])
    >>> similarities = cn.predict([0, 1])
    >>> similarities
    array([[2, 1, 1, 1, 1],
           [1, 3, 0, 2, 1]])
    >>> similarities = cn.predict((0, 1))
    >>> similarities
    1
    >>> similarities = cn.predict([(0, 1), (1, 2)])
    >>> similarities
    array([1, 0])
    """
    def __init__(self):
        super(CommonNeighbors, self).__init__()

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]):
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph

        Returns
        -------
        self : :class:`CommonNeighbors`
        """
        adjacency = check_format(adjacency)
        self.indptr_ = adjacency.indptr
        self.indices_ = adjacency.indices
        return self

    def _predict_node(self, node: int):
        """Prediction for a single node."""
        n_row = self.indptr_.shape[0] - 1
        neigh_i = self._neighborhood(node)

        preds = np.zeros(n_row, dtype=int)
        for j in range(n_row):
            neigh_j = self._neighborhood(j)
            preds[j] = np.intersect1d(neigh_i, neigh_j, assume_unique=True).shape[0]
        return preds

    def _predict_edge(self, source: int, target: int):
        """Prediction for a single edge."""
        neigh_i = self._neighborhood(source)
        neigh_j = self._neighborhood(target)
        return np.intersect1d(neigh_i, neigh_j, assume_unique=True).shape[0]
