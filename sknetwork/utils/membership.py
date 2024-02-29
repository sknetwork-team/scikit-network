#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Optional

import numpy as np
from scipy import sparse

from sknetwork.utils.neighbors import get_degrees


def get_membership(labels: np.ndarray, dtype=bool, n_labels: Optional[int] = None) -> sparse.csr_matrix:
    """Build the binary matrix of the label assignments, of shape n_samples x n_labels.
    Negative labels are ignored.

    Parameters
    ----------
    labels :
        Label of each node (integers).
    dtype :
        Type of the output. Boolean by default.
    n_labels : int
        Number of labels.

    Returns
    -------
    membership : sparse.csr_matrix
        Binary matrix of label assignments.

    Example
    -------
    >>> from sknetwork.utils import get_membership
    >>> labels = np.array([0, 0, 1, 2])
    >>> membership = get_membership(labels)
    >>> membership.toarray().astype(int)
    array([[1, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]])
    """
    n: int = len(labels)
    if n_labels is None:
        shape = (n, max(labels)+1)
    else:
        shape = (n, n_labels)
    ix = (labels >= 0)
    data = np.ones(ix.sum())
    row = np.arange(n)[ix]
    col = labels[ix]
    return sparse.csr_matrix((data, (row, col)), shape=shape, dtype=dtype)


def from_membership(membership: sparse.csr_matrix) -> np.ndarray:
    """Get the labels from a membership matrix (n_samples x n_labels).
    Samples without label get -1.

    Parameters
    ----------
    membership :
        Membership matrix.

    Returns
    -------
    labels : np.ndarray
        Labels (columns indices of the membership matrix).
    Example
    -------
    >>> from scipy import sparse
    >>> from sknetwork.utils import from_membership
    >>> membership = sparse.eye(3).tocsr()
    >>> labels = from_membership(membership)
    >>> labels
    array([0, 1, 2])
    """
    mask = get_degrees(membership) > 0
    labels = -np.ones(membership.shape[0], dtype=int)
    labels[mask] = membership.indices
    return labels
