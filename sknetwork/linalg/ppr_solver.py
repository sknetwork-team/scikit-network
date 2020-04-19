#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs, LinearOperator, bicgstab

from sknetwork.linalg.diteration import diffusion
from sknetwork.linalg.normalization import normalize


class RandomSurferOperator(LinearOperator):
    """Random surfer as a LinearOperator

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph as a CSR or a LinearOperator.
    damping_factor : float
        Probability to continue the random walk.
    seeds :
        Probability vector for seeds.

    Attributes
    ----------
    a : sparse.csr_matrix
        Scaled transposed transition matrix.
    b : np.ndarray
        Scaled restart probability vector.
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, LinearOperator], seeds: np.ndarray, damping_factor):
        super(RandomSurferOperator, self).__init__(shape=adjacency.shape, dtype=float)

        n = adjacency.shape[0]
        out_degrees = adjacency.dot(np.ones(n))
        damping_matrix = damping_factor * sparse.eye(n, format='csr')

        if hasattr(adjacency, 'left_sparse_dot'):
            self.a = normalize(adjacency).left_sparse_dot(damping_matrix).T
        else:
            self.a = (damping_matrix.dot(normalize(adjacency))).T.tocsr()
        self.b = (np.ones(n) - damping_factor * out_degrees.astype(bool)) * seeds

    def _matvec(self, x: np.ndarray):
        return self.a.dot(x) + self.b * x.sum()


def get_pagerank(adjacency: Union[sparse.csr_matrix, LinearOperator], seeds: np.ndarray, damping_factor: float,
                 n_iter: int, tol: float = 0., solver: str = 'naive'):
    """Pagerank."""
    n = adjacency.shape[0]

    if solver == 'diteration':
        if not isinstance(adjacency, sparse.csr_matrix):
            raise ValueError('D-iteration is not compatible with linear operators.')
        adjacency = normalize(adjacency, p=1)
        indptr = adjacency.indptr
        indices = adjacency.indices
        data = adjacency.data

        scores = np.zeros(n)
        fluid = (1 - damping_factor) * seeds
        scores = diffusion(indptr, indices, data, scores, fluid, damping_factor, n_iter)

    else:
        rso = RandomSurferOperator(adjacency, seeds, damping_factor)

        if solver == 'bicgstab':
            scores, info = bicgstab(sparse.eye(n, format='csr') - rso.a, rso.b, atol=tol)
        elif solver == 'lanczos':
            # noinspection PyTypeChecker
            _, scores = sparse.linalg.eigs(rso, k=1, tol=tol)
        elif solver == 'naive':
            scores = rso.b
            for i in range(n_iter):
                scores = rso.dot(scores)
                scores /= scores.sum()
        else:
            raise ValueError('Unknown solver.')

        scores = abs(scores.flatten().real)
    return scores / scores.sum()
