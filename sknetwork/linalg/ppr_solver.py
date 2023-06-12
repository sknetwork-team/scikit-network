#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs, LinearOperator, bicgstab

from sknetwork.linalg.diteration import diffusion
from sknetwork.linalg.push import push_pagerank
from sknetwork.linalg.normalizer import normalize
from sknetwork.linalg.polynome import Polynome


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
        out_degrees = adjacency.dot(np.ones(n)).astype(bool)

        if hasattr(adjacency, 'left_sparse_dot'):
            self.a = damping_factor * normalize(adjacency).T
        else:
            self.a = (damping_factor * normalize(adjacency)).T.tocsr()
        self.b = (np.ones(n) - damping_factor * out_degrees) * seeds

    def _matvec(self, x: np.ndarray):
        return self.a.dot(x) + self.b * x.sum()


def get_pagerank(adjacency: Union[sparse.csr_matrix, LinearOperator], seeds: np.ndarray, damping_factor: float,
                 n_iter: int, tol: float = 1e-6, solver: str = 'piteration') -> np.ndarray:
    """Solve the Pagerank problem. Formally,

    :math:`x = \\alpha Px + (1-\\alpha)y`,

    where :math:`P = (D^{-1}A)^T` is the transition matrix and :math:`y` is the personalization probability vector.

    Parameters
    ----------
    adjacency : sparse.csr_matrix
        Adjacency matrix of the graph.
    seeds : np.ndarray
        Personalization array. Must be a valid probability vector.
    damping_factor : float
        Probability to continue the random walk.
    n_iter : int
        Number of iterations for some of the solvers such as ``'piteration'`` or ``'diteration'``.
    tol : float
        Tolerance for the convergence of some solvers such as ``'bicgstab'`` or ``'lanczos'`` or ``'push'``.
    solver : :obj:`str`
        Which solver to use: ``'piteration'``, ``'diteration'``, ``'bicgstab'``, ``'lanczos'``, ``̀'RH'``, ``'push'``.

    Returns
    -------
    pagerank : np.ndarray
        Probability vector.

    Examples
    --------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> seeds = np.array([1, 0, 0, 0, 0])
    >>> scores = get_pagerank(adjacency, seeds, damping_factor=0.85, n_iter=10)
    >>> np.round(scores, 2)
    array([0.29, 0.24, 0.12, 0.12, 0.24])


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
    * Whang, J. , Lenharth, A. , Dhillon, I. , & Pingali, K. . (2015).
      `Scalable Data-Driven PageRank: Algorithms, System Issues, and Lessons Learned`. 9233, 438-450.
      <https://www.cs.utexas.edu/users/inderjit/public_papers/scalable_pagerank_europar15.pdf>
    """
    n = adjacency.shape[0]

    if solver == 'diteration':
        if not isinstance(adjacency, sparse.csr_matrix):
            raise ValueError('D-iteration is not compatible with linear operators.')
        adjacency = normalize(adjacency, p=1)
        indptr = adjacency.indptr.astype(np.int32)
        indices = adjacency.indices.astype(np.int32)
        data = adjacency.data.astype(np.float32)
        damping_factor = np.float32(damping_factor)
        n_iter = np.int32(n_iter)
        tol = np.float32(tol)

        scores = np.zeros(n, dtype=np.float32)
        fluid = (1 - damping_factor) * seeds.astype(np.float32)
        diffusion(indptr, indices, data, scores, fluid, damping_factor, n_iter, tol)

    elif solver == 'push':
        n = adjacency.shape[0]
        damping_factor = np.float32(damping_factor)
        tol = np.float32(tol)
        degrees = adjacency.dot(np.ones(n)).astype(np.int32)
        rev_adjacency = adjacency.transpose().tocsr()

        indptr = adjacency.indptr.astype(np.int32)
        indices = adjacency.indices.astype(np.int32)
        rev_indptr = rev_adjacency.indptr.astype(np.int32)
        rev_indices = rev_adjacency.indices.astype(np.int32)

        scores = push_pagerank(n, degrees, indptr, indices,
                               rev_indptr, rev_indices,
                               seeds.astype(np.float32),
                               damping_factor, tol)

    elif solver == 'RH':
        coeffs = np.ones(n_iter + 1)
        polynome = Polynome(damping_factor * normalize(adjacency, p=1).T.tocsr(), coeffs)
        scores = polynome.dot(seeds)

    else:
        rso = RandomSurferOperator(adjacency, seeds, damping_factor)
        v0 = rso.b
        if solver == 'bicgstab':
            scores, info = bicgstab(sparse.eye(n, format='csr') - rso.a, rso.b, atol=tol, x0=v0)
        elif solver == 'lanczos':
            # noinspection PyTypeChecker
            _, scores = sparse.linalg.eigs(rso, k=1, tol=tol, v0=v0)
            scores = abs(scores.flatten().real)
        elif solver == 'piteration':
            scores = v0
            for i in range(n_iter):
                scores_ = rso.dot(scores)
                scores_ /= scores_.sum()
                if np.linalg.norm(scores - scores_, ord=1) < tol:
                    break
                else:
                    scores = scores_
        else:
            raise ValueError('Unknown solver.')

    return scores / scores.sum()
