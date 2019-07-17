#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 4 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sknetwork.utils.checks import check_format, is_square
from sknetwork.utils.adjacency_formats import bipartite2undirected
from typing import Union


def is_connected(adjacency: sparse.csr_matrix) -> bool:
    """
    Check whether a graph is weakly connected. Bipartite graphs are treated as undirected ones.

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph.

    Returns
    -------

    """
    n, m = adjacency.shape
    if n != m:
        adjacency = bipartite2undirected(adjacency)
    return connected_components(adjacency, directed=False)[0] == 1


def largest_connected_component(adjacency: Union[sparse.csr_matrix, np.ndarray], return_labels: bool = False):
    """
    Extract largest connected component of a graph. Bipartite graphs are treated as undirected ones.

    Parameters
    ----------
    adjacency
        Adjacency or biadjacency matrix of the graph.
    return_labels: bool
        Whether to return the indices of the new nodes in the original graph.

    Returns
    -------
    new_adjacency: sparse.csr_matrix
        Adjacency or biadjacency matrix of the largest connected component.
    indices: array or tuple of array
        Indices of the nodes in the original graph. For biadjacency matrices,
        ``indices[0]`` corresponds to the rows and ``indices[1]`` to the columns.

    """
    adjacency = check_format(adjacency)
    n_samples, n_features = adjacency.shape
    if not is_square(adjacency):
        bipartite: bool = True
        full_adjacency = sparse.bmat([[None, adjacency], [adjacency.T, None]], format='csr')
    else:
        bipartite: bool = False
        full_adjacency = adjacency

    n_components, labels = connected_components(full_adjacency)
    unique_labels, counts = np.unique(labels, return_counts=True)
    component_label = unique_labels[np.argmax(counts)]
    component_indices = np.where(labels == component_label)[0]

    if bipartite:
        split_ix = np.searchsorted(component_indices, n_samples)
        samples_ix, features_ix = component_indices[:split_ix], component_indices[split_ix:] - n_samples
    else:
        samples_ix, features_ix = component_indices, component_indices
    new_adjacency = adjacency[samples_ix, :]
    new_adjacency = (new_adjacency.tocsc()[:, features_ix]).tocsr()

    if return_labels:
        if bipartite:
            return new_adjacency, (samples_ix, features_ix)
        else:
            return new_adjacency, samples_ix
    else:
        return new_adjacency
