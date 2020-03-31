#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 29, 2018
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""

import numpy as np
from scipy import sparse

from sknetwork.data.base import BaseGraph
from sknetwork.data.base_bipartite import BaseBiGraph


class Small(BaseGraph):
    """Small graph (undirected), used for testing.
    10 nodes, 10 edges.
    No label.
    """

    def __init__(self):
        super(Small, self).__init__()

        row = np.array([0, 1, 3, 2, 6, 6, 6, 7, 8, 9])
        col = np.array([1, 3, 2, 5, 4, 5, 7, 9, 9, 9])
        data = np.array([1, 2.5, 1, 2, 2, 1, 2, 2, 1.5, 2])
        adjacency = sparse.csr_matrix((data, (row, col)), shape=(10, 10))
        self.adjacency = adjacency + adjacency.T


class DiSmall(BaseGraph):
    """Small graph (directed), used for testing.
    10 nodes, 10 edges
    No label.
    """

    def __init__(self):
        super(DiSmall, self).__init__()

        row = np.array([0, 1, 3, 4, 6, 6, 6, 7, 8, 9])
        col = np.array([1, 3, 2, 5, 4, 5, 7, 9, 9, 9])
        data = np.array([1, 2.5, 1, 2, 2, 1, 2, 2, 1.5, 2])
        adjacency = sparse.csr_matrix((data, (row, col)), shape=(10, 10))
        self.adjacency = adjacency


class BiSmall(BaseBiGraph):
    """Small graph (bipartite), used for testing.
    4 + 5 nodes, 6 edges.
    No label.
    """

    def __init__(self):
        super(BiSmall, self).__init__()

        row = np.array([0, 1, 1, 2, 2, 3])
        col = np.array([1, 2, 3, 1, 0, 4])
        data = np.array([1, 2.5, 1, 2, 2, 1.5])
        biadjacency = sparse.csr_matrix((data, (row, col)), shape=(4, 5))
        self.biadjacency = biadjacency


class SmallDisconnected(BaseGraph):
    """Small graph (undirected), used for testing.
    10 nodes, 10 edges.
    No label.
    """

    def __init__(self):
        super(SmallDisconnected, self).__init__()

        row = np.array([1, 2, 3, 4, 6, 6, 6, 7, 8, 9])
        col = np.array([1, 3, 2, 5, 4, 5, 7, 9, 9, 9])
        data = np.array([1, 2.5, 1, 2, 2, 1, 2, 2, 1.5, 2])
        adjacency = sparse.csr_matrix((data, (row, col)), shape=(10, 10))
        self.adjacency = adjacency + adjacency.T


class BiSmallDisconnected(BaseBiGraph):
    """Small graph (bipartite), used for testing.
    4 + 5 nodes, 6 edges.
    No label.
    """

    def __init__(self):
        super(BiSmallDisconnected, self).__init__()

        row = np.array([1, 1, 1, 2, 2, 3])
        col = np.array([1, 2, 3, 1, 3, 4])
        data = np.array([1, 2.5, 1, 2, 2, 1.5])
        biadjacency = sparse.csr_matrix((data, (row, col)), shape=(4, 5))
        self.biadjacency = biadjacency
