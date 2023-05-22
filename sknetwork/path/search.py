#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in May 2023
"""
import numpy as np
from scipy import sparse

from sknetwork.path.distances import get_distances


def breadth_first_search(adjacency: sparse.csr_matrix, source: int):
    """Breadth-first ordering starting from some node.

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    source : int
        Source node.

    Returns
    -------
    index : np.ndarray
        Node index corresponding to the breadth-first-search from the source.
        The length of the vector is the number of nodes reachable from the source.
    """
    distances = get_distances(adjacency, source)
    indices = np.argsort(distances)
    n = np.sum(distances < 0)
    return indices[n:]
