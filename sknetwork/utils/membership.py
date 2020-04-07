#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 10, 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

import numpy as np
from scipy import sparse


def membership_matrix(labels: np.ndarray, dtype=bool) -> sparse.csr_matrix:
    """
    Build a n x k matrix of the label assignments, with k the number of labels.
    Negative labels are ignored.

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
    shape = (n, max(labels)+1)
    ix = (labels >= 0)
    data = np.ones(ix.sum())
    row = np.arange(n)[ix]
    col = labels[ix]
    return sparse.csr_matrix((data, (row, col)), shape=shape, dtype=dtype)
