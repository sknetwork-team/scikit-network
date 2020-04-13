#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 10, 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Optional

import numpy as np
from scipy import sparse


def membership_matrix(labels: np.ndarray, dtype=bool, n_labels: Optional[int] = None) -> sparse.csr_matrix:
    """Build a n x k matrix of the label assignments, with k the number of labels.
    Negative labels are ignored.

    Parameters
    ----------
    labels :
        Label of each node.
    dtype :
        Type of the entries. Boolean by default.
    n_labels : int
        Number of labels.

    Returns
    -------
    membership : sparse.csr_matrix
        Binary matrix of label assignments.

    Notes
    -----
    The inverse operation is simply ``labels = membership.indices``.
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
