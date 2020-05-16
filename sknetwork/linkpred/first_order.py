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

    def neighborhood(self, node: int):
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
    >>> commonneigh = CommonNeighbors()
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

    def predict_node(self, node: int):
        """Prediction for a single node."""
        n_row = self.indptr_.shape[0] - 1
        neigh_i = self.neighborhood(node)

        preds = np.zeros(n_row)
        for j in range(n_row):
            neigh_j = self.neighborhood(j)
            preds[j] = len(set(neigh_i) & set(neigh_j))
        return preds

    def predict_edge(self, source: int, target: int):
        """Prediction for a single edge."""
        neigh_i = self.neighborhood(source)
        neigh_j = self.neighborhood(target)
        return len(set(neigh_i) & set(neigh_j))
