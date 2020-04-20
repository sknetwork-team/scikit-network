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
    """Solve the Pagerank problem.

    References
    ----------
    * Hong, D. (2012). `Optimized on-line computation of pagerank algorithm.
      <https://arxiv.org/pdf/1202.6158.pdf>`_
      arXiv preprint arXiv:1202.6158.
    * Van der Vorst, H. A. (1992). `Bi-CGSTAB:
      <https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method>`_
      A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems.
      SIAM Journal on scientific and Statistical Computing, 13(2), 631-644.
    * Lanczos, C. (1950).
      `An iteration method for the solution of the eigenvalue problem of linear differential and integral operators.
      <http://www.cs.umd.edu/~oleary/lanczos1950.pdf>`_
      Los Angeles, CA: United States Governm. Press Office.
    """
    n = adjacency.shape[0]

    if solver == 'diteration':
        if not isinstance(adjacency, sparse.csr_matrix):
            raise ValueError('D-iteration is not compatible with linear operators.')
        adjacency = normalize(adjacency, p=1)
        indptr = adjacency.indptr.astype(np.int32)
        indices = adjacency.indices.astype(np.int32)
        data = adjacency.data.astype(np.float32)
        n_iter = np.int32(n_iter)
        damping_factor = np.float32(damping_factor)

        scores = np.zeros(n, dtype=np.float32)
        fluid = (1 - damping_factor) * seeds.astype(np.float32)
        scores = diffusion(indptr, indices, data, scores, fluid, damping_factor, n_iter)

    else:
        rso = RandomSurferOperator(adjacency, seeds, damping_factor)
        v0 = rso.b
        if solver == 'bicgstab':
            scores, info = bicgstab(sparse.eye(n, format='csr') - rso.a, rso.b, atol=tol, x0=v0)
        elif solver == 'lanczos':
            # noinspection PyTypeChecker
            _, scores = sparse.linalg.eigs(rso, k=1, tol=tol, v0=v0)
        elif solver == 'naive':
            scores = v0
            for i in range(n_iter):
                scores = rso.dot(scores)
                scores /= scores.sum()
        else:
            raise ValueError('Unknown solver.')

        scores = abs(scores.flatten().real)
    return scores / scores.sum()
