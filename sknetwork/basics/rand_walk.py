#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from sknetwork.linalg import diag_pinv


def transition_matrix(adjacency: Union[sparse.csr_matrix, np.ndarray, LinearOperator]):
    """Compute the transition matrix of the random walk :

    :math:`P = D^+A`,

    where :math:`D^+` is the pseudo-inverse of the degree matrix.

    Parameters
    ----------
    adjacency :
        Adjacency or biadjacency matrix.

    Returns
    -------
    Transition matrix as a CSR or a LinearOperator

    """
    if isinstance(adjacency, np.ndarray):
        adjacency = sparse.csr_matrix(adjacency)
    d: np.ndarray = adjacency.dot(np.ones(adjacency.shape[1]))

    if hasattr(adjacency, 'left_sparse_dot'):
        return adjacency.left_sparse_dot(diag_pinv(d))
    return diag_pinv(d).dot(adjacency)
