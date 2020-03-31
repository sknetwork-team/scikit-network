#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2020
@author: Thomas Bonald <tbonald@enst.fr>
"""

import numpy as np


class Graph:
    """Graph.

    Attributes
    ----------
    adjacency : sparse.csr_matrix, shape (n, n)
        Adjacency matrix of the graph.
    labels : np.ndarray, shape (n,)
        Labels of nodes.
    names : np.ndarray, shape (n,)
        Names of nodes.
    labels_name : dict
        Names of labels.
    pos : np.ndarray, shape (n, 2)
        Position of nodes for visualization.
    """

    def __init__(self):
        self.adjacency = None
        self.labels = None
        self.names = None
        self.labels_name = None
        self.pos = None


class BiGraph:
    """Bipartite graph.

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
    names : np.ndarray, shape (n1,)
        Names of rows.
    names_row : np.ndarray, shape (n1,)
        Names of rows (copy of **names**).
    names_col : np.ndarray, shape (n2,)
        Names of columns.
    labels_name : dict
        Names of labels on rows.
    labels_row_name : dict
        Names of labels on rows (copy of **names_labels**).
    labels_col_name : dict
        Names of labels on columns.
    pos_row : np.ndarray, shape (n1, 2)
        Position of rows for visualization.
    pos_col : np.ndarray, shape (n2, 2)
        Position of columns.
    """

    def __init__(self):
        super(BiGraph, self).__init__()

        self.biadjacency = None
        self.labels = None
        self.labels_row = None
        self.labels_col = None
        self.names = None
        self.names_row = None
        self.names_col = None
        self.labels_name = None
        self.labels_row_name = None
        self.labels_col_name = None
        self.pos_row = None
        self.pos_col = None

    @staticmethod
    def _get_position(self):
        n1, n2 = self.biadjacency.shape
        pos_row = np.vstack((np.zeros(n1), np.arange(n1))).T
        pos_col = np.vstack((np.ones(n2), np.arange(n2))).T
        return pos_row, pos_col
