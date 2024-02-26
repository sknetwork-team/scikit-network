#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 10, 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Optional

import numpy as np
from scipy import sparse

from sknetwork.utils.membership import get_membership


def reindex_labels(labels: np.ndarray) -> np.ndarray:
    """Reindex clusters in decreasing order of size.

    Parameters
    ----------
    labels :
        Label of each node.
    Returns
    -------
    new_labels : np.ndarray
        New label of each node.

    Example
    -------
    >>> from sknetwork.clustering import reindex_labels
    >>> labels = np.array([0, 1, 1])
    >>> reindex_labels(labels)
    array([1, 0, 0])
    """
    _, index, counts = np.unique(labels, return_inverse=True, return_counts=True)
    _, new_index = np.unique(np.argsort(-counts), return_index=True)
    return new_index[index]


def aggregate_graph(input_matrix: sparse.csr_matrix, labels: Optional[np.ndarray] = None,
                    labels_row: Optional[np.ndarray] = None, labels_col: Optional[np.ndarray] = None) \
        -> sparse.csr_matrix:
    """Aggregate graph per label. All nodes with the same label become a single node.
    Negative labels are ignored (corresponding nodes are discarded).

    Parameters
    ----------
    input_matrix: sparse matrix
        Adjacency or biadjacency matrix of the graph.
    labels: np.ndarray
        Labels of nodes.
    labels_row: np.ndarray
        Labels of rows (for bipartite graphs). Alias for labels.
    labels_col: np.ndarray
        Labels of columns (for bipartite graphs).
    """
    if labels_row is not None:
        membership_row = get_membership(labels_row)
    else:
        membership_row = get_membership(labels)
    if labels_col is not None:
        membership_col = get_membership(labels_col)
    else:
        membership_col = membership_row
    aggregate_matrix = membership_row.T.dot(input_matrix).dot(membership_col)
    return aggregate_matrix.tocsr()
