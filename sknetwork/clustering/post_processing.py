#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 10, 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import numpy as np
from scipy import sparse


def membership_matrix(labels: np.ndarray, dtype=bool) -> sparse.csr_matrix:
    """
    Builds a n x k matrix of the label assignments, with k the number of labels.

    Parameters
    ----------
    labels :
        Label of each node.
    dtype :
        Type of the entries. Boolean by default.

    Returns
    -------
    membership :
        Binary matrix of label assignments.

    """
    n_nodes = len(labels)
    return sparse.csr_matrix((np.ones(n_nodes), (np.arange(n_nodes), labels)), dtype=dtype)


def reindex_clusters(labels: np.ndarray) -> np.ndarray:
    """
    Reindex clusters in decreasing order of size.

    Parameters
    ----------
    labels:
        label of each node.

    Returns
    -------
    new_labels:
        new label of each node.

    """
    unique_values, counts = np.unique(labels, return_counts=True)
    sorted_values = unique_values[np.argsort(-counts)]
    new_index = {l: i for i, l in enumerate(sorted_values)}
    new_labels = np.array([new_index[l] for l in labels])
    return new_labels
