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


def test_graph():
    """Simple undirected graph, used for testing.
    10 nodes, 10 edges.
    """
    row = np.array([0, 1, 3, 2, 6, 6, 6, 7, 8, 9])
    col = np.array([1, 3, 2, 5, 4, 5, 7, 9, 9, 9])
    data = np.array([1, 2.5, 1, 2, 2, 1, 2, 2, 1.5, 2])
    adjacency = sparse.csr_matrix((data, (row, col)), shape=(10, 10))
    return adjacency + adjacency.T


def test_digraph():
    """Simple directed graph, used for testing.
    10 nodes, 10 edges
    """
    row = np.array([0, 1, 3, 4, 6, 6, 6, 7, 8, 9])
    col = np.array([1, 3, 2, 5, 4, 5, 7, 9, 9, 9])
    data = np.array([1, 2.5, 1, 2, 2, 1, 2, 2, 1.5, 2])
    adjacency = sparse.csr_matrix((data, (row, col)), shape=(10, 10))
    return adjacency


def test_bigraph():
    """Simple bipartite graph, used for testing.
    4 + 5 nodes, 6 edges.
    """
    row = np.array([0, 1, 1, 2, 2, 3])
    col = np.array([1, 2, 3, 1, 0, 4])
    data = np.array([1, 2.5, 1, 2, 2, 1.5])
    biadjacency = sparse.csr_matrix((data, (row, col)), shape=(4, 5))
    return biadjacency


def test_graph_disconnect():
    """Simple disconnected undirected graph, used for testing.
    10 nodes, 10 edges.
    """
    row = np.array([1, 2, 3, 4, 6, 6, 6, 7, 8, 9])
    col = np.array([1, 3, 2, 5, 4, 5, 7, 9, 9, 9])
    data = np.array([1, 2.5, 1, 2, 2, 1, 2, 2, 1.5, 2])
    adjacency = sparse.csr_matrix((data, (row, col)), shape=(10, 10))
    return adjacency + adjacency.T


def test_bigraph_disconnect():
    """Simple disconnected bipartite graph, used for testing.
    4 + 5 nodes, 6 edges.
    """
    row = np.array([1, 1, 1, 2, 2, 3])
    col = np.array([1, 2, 3, 1, 3, 4])
    data = np.array([1, 2.5, 1, 2, 2, 1.5])
    biadjacency = sparse.csr_matrix((data, (row, col)), shape=(4, 5))
    return biadjacency
