#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 10, 2019
@author: Thomas Bonald <bonald@enst.fr>, Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse


def membership_matrix(labels: np.ndarray, dtype=bool) -> sparse.csr_matrix:
    """
    Build a n x k matrix of the label assignments, with k the number of labels.

    Parameters
    ----------
    labels :
        Label of each node.
    dtype :
        Type of the entries. Boolean by default.

    Returns
    -------
    membership : sparse.csr_matrix
        Binary matrix of label assignments.

    Notes
    -----
    The inverse operation is simply ``labels = membership.indices``.

    """
    n: int = len(labels)
    return sparse.csr_matrix((np.ones(n), (np.arange(n), labels)), dtype=dtype)


def reindex_clusters(labels: np.ndarray) -> np.ndarray:
    """
    Reindex clusters in decreasing order of size.

    Parameters
    ----------
    labels:
        label of each node.

    Returns
    -------
    new_labels: np.ndarray
        new label of each node.

    """
    _, labels = np.unique(labels, return_inverse=True)
    unique_values, counts = np.unique(labels, return_counts=True)
    sorted_values = unique_values[np.argsort(-counts)]
    _, new_index = np.unique(sorted_values, return_index=True)
    new_labels = new_index[labels.astype(int)]
    return new_labels


def test_runtime(n):
    from time import time
    labels = np.random.choice(100, n)
    t0 = time()
    reindex_clusters(labels)
    delta = time() - t0
    return delta
