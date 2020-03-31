#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2020
@author: Thomas Bonald <tbonald@enst.fr>
"""
import numpy as np

from sknetwork.data.base import BaseGraph


class BaseBiGraph(BaseGraph):
    """Base class for bipartite graphs.

    Attributes
    ----------
    biadjacency : sparse.csr_matrix, shape (n1, n2)
        Biadjacency matrix of the graph.
    labels : np.ndarray, shape (n1,)
        Labels of rows.
    labels_row : np.ndarray, shape (n1,)
        Labels of rows (copy of **labels**).
    labels_col : np.ndarray, shape (n2,)
        Labels of columns.
    names : np.ndarray, shape (n,)
        Names of rows.
    names_row : np.ndarray, shape (n1,)
        Names of rows (copy of **names**).
    names_col : np.ndarray, shape (n2,)
        Names of columns.
    pos_row : np.ndarray, shape (n1, 2)
        Position of rows for visualization.
    pos_col : np.ndarray, shape (n2, 2)
        Position of columns.
    """

    def __init__(self):
        super(BaseBiGraph, self).__init__()

        self.biadjacency = None
        self.labels = None
        self.labels_row = None
        self.labels_col = None
        self.names = None
        self.names_row = None
        self.names_col = None
        self.pos_row = None
        self.pos_col = None

    @staticmethod
    def _get_position(self):
        n1, n2 = self.biadjacency.shape
        pos_row = np.vstack((np.zeros(n1), np.arange(n1))).T
        pos_col = np.vstack((np.ones(n2), np.arange(n2))).T
        return pos_row, pos_col

