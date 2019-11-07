#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.utils.checks import check_format


def transition_matrix(adjacency: Union[sparse.csr_matrix, np.ndarray]):
    """Compute

    :math:`P = D^+A`,

    where :math:`D^+` is the pseudo-inverse of the degree matrix.

    Parameters
    ----------
    adjacency

    Returns
    -------
    P:
        The transition matrix.

    """
    adjacency = sparse.csr_matrix(adjacency)

    d: np.ndarray = adjacency.dot(np.ones(adjacency.shape[1]))
    diag_out: sparse.csr_matrix = sparse.diags(d, format='csr')
    diag_out.data = 1 / diag_out.data

    return diag_out.dot(adjacency)
