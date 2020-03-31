#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2020
@author: Thomas Bonald <tbonald@enst.fr>
"""
from abc import ABC


class BaseGraph(ABC):
    """Base class for graphs.

    Attributes
    ----------
    adjacency : sparse.csr_matrix, shape (n, n)
        Adjacency matrix of the graph.
    labels : np.ndarray, shape (n,)
        Labels of nodes.
    names : np.ndarray, shape (n,)
        Names of nodes.
    pos : np.ndarray, shape (n, 2)
        Position of nodes for visualization.
    """

    def __init__(self):
        self.adjacency = None
        self.labels = None
        self.names = None
        self.pos = None

