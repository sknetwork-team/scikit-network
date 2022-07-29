#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2022
@author: Thomas Bonald <thomas.bonald@telecom-paris.fr>
"""

import numpy as np
from scipy import sparse


def get_laplacian(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Return the Laplacian matrix of a graph."""
    weights = adjacency.dot(np.ones(adjacency.shape[0]))
    return sparse.diags(weights) - adjacency
