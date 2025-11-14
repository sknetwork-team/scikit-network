#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 29, 2018
@author: Quentin Lutz <qlutz@enst.fr>
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <tbonald@enst.fr>
"""

import numpy as np
from scipy import sparse
import pytest

from sknetwork.utils import directed2undirected


@pytest.mark.skip
def test_digraph() -> sparse.csr_matrix:
    """Simple directed graph, used for testing.
    10 nodes, 12 edges
    """
    row = np.array([0, 1, 1, 3, 4, 6, 6, 6, 6, 7, 8, 8, 9])
    col = np.array([1, 4, 3, 2, 5, 4, 5, 7, 1, 9, 9, 2, 9])
    data = np.array([1, 1, 2.5, 1, 2, 2, 1, 2, 2, 1.5, 2, 1, 2])
    return sparse.csr_matrix((data, (row, col)), shape=(10, 10))


@pytest.mark.skip
def test_graph() -> sparse.csr_matrix:
    """Simple undirected graph, used for testing.
    10 nodes, 12 edges.
    """
    return directed2undirected(test_digraph(), weighted=True)


@pytest.mark.skip
def test_bigraph() -> sparse.csr_matrix:
    """Simple bipartite graph, used for testing.
    6 + 8 nodes, 9 edges.
    """
    row = np.array([0, 1, 1, 2, 2, 3, 4, 5, 5])
    col = np.array([1, 2, 3, 1, 0, 4, 7, 5, 6])
    data = np.array([1, 2.5, 1, 2, 2, 1.5, 1, 2, 3])
    return sparse.csr_matrix((data, (row, col)), shape=(6, 8))


@pytest.mark.skip
def test_disconnected_graph() -> sparse.csr_matrix:
    """Simple disconnected undirected graph, used for testing.
    10 nodes, 10 edges.
    """
    row = np.array([1, 2, 3, 4, 6, 6, 6, 7, 8, 9])
    col = np.array([1, 3, 2, 5, 4, 5, 7, 9, 9, 9])
    data = np.array([1, 2.5, 1, 2, 2, 1, 2, 2, 1.5, 2])
    adjacency = sparse.csr_matrix((data, (row, col)), shape=(10, 10))
    return directed2undirected(adjacency)


@pytest.mark.skip
def test_bigraph_disconnect() -> sparse.csr_matrix:
    """Simple disconnected bipartite graph, used for testing.
    6 + 8 nodes, 9 edges.
    """
    row = np.array([1, 1, 1, 2, 2, 3, 5, 4, 5])
    col = np.array([1, 2, 3, 1, 3, 4, 7, 7, 6])
    data = np.array([1, 2.5, 1, 2, 2, 1.5, 3, 0, 1])
    return sparse.csr_matrix((data, (row, col)), shape=(6, 8))


@pytest.mark.skip
def test_graph_bool() -> sparse.csr_matrix:
    """Simple undirected graph with boolean entries, used for testing (10 nodes, 10 edges)."""
    adjacency = test_graph()
    adjacency.data = adjacency.data.astype(bool)
    return adjacency


@pytest.mark.skip
def test_clique() -> sparse.csr_matrix:
    """Clique graph, used for testing (10 nodes, 45 edges).
    """
    n = 10
    adjacency = sparse.csr_matrix(np.ones((n, n), dtype=bool))
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


@pytest.mark.skip
def test_graph_empty() -> sparse.csr_matrix:
    """Empty graph, used for testing (10 nodes, 0 edges).
    """
    return sparse.csr_matrix((10, 10), dtype=bool)
