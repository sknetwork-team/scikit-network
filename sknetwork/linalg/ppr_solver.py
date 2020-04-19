#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
import numpy as np
from scipy import sparse

from sknetwork.linalg.diteration import diffusion
from sknetwork.linalg.normalization import normalize


def pagerank(adjacency: sparse.csr_matrix, seeds, damping_factor: float = 0.85, n_iter: int = 10):
    """Pagerank by D-iteration."""
    n = adjacency.shape[0]
    adjacency = normalize(adjacency, p=1)
    indptr = adjacency.indptr
    indices = adjacency.indices
    data = adjacency.data

    scores = np.zeros(n)
    fluid = (1 - damping_factor) * seeds
    for i in range(n_iter):
        fluid, scores = diffusion(indptr, indices, data, scores, fluid, damping_factor)
    return scores / scores.sum()
