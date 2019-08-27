#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 31 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

from typing import Union, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs, LinearOperator, lsqr, spsolve

from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format, has_nonnegative_entries, is_square
from sknetwork.utils.adjacency_formats import bipartite2undirected


def restart_probability(n: int, personalization: Union[dict, np.ndarray] = None) -> np.ndarray:
    """

    Parameters
    ----------
    n :
        Total number of samples.
    personalization :
        If ``None``, the uniform distribution is used.
        Otherwise, a non-negative, non-zero vector or a dictionary must be provided.

    Returns
    -------
    restart_prob:
        A probability vector.

    """
    if personalization is None:
        restart_prob: np.ndarray = np.ones(n) / n
    else:
        if type(personalization) == dict:
            tmp = np.zeros(n)
            tmp[list(personalization.keys())] = list(personalization.values())
            personalization = tmp
        if type(personalization) == np.ndarray and len(personalization) == n \
           and has_nonnegative_entries(personalization) and np.sum(personalization):
            restart_prob = personalization.astype(float) / np.sum(personalization)
        else:
            raise ValueError('Personalization must be None or a non-negative, non-null vector or a dictionary.')
    return restart_prob


class RandomSurferOperator(LinearOperator):
    """
    Random surfer as a LinearOperator

    Parameters
    ----------
    adjacency :
        Adjacency matrix of the graph.
    damping_factor : float
        Probability to continue the random walk.
    personalization :
        If ``None``, the uniform distribution is used.
        Otherwise, a non-negative, non-zero vector or a dictionary must be provided.

    Attributes
    ----------
    a : sparse.csr_matrix
        Scaled transposed transition matrix.
    b : np.ndarray
        Scaled restart probability vector.

    """
    def __init__(self, adjacency: sparse.csr_matrix, damping_factor: float = 0.85, personalization=None):
        LinearOperator.__init__(self, shape=adjacency.shape, dtype=float)
        n = adjacency.shape[0]

        out_degrees = adjacency.dot(np.ones(n))
        diag_out = sparse.diags(out_degrees, format='csr')
        diag_out.data = 1. / diag_out.data
        self.a = damping_factor * adjacency.T.dot(diag_out)

        restart_prob = restart_probability(n, personalization)
        self.b = (np.ones(n) - damping_factor * out_degrees.astype(bool)) * restart_prob

    def _matvec(self, x):
        return self.a.dot(x) + self.b * x.sum()


class PageRank(Algorithm):
    """
    Computes the PageRank of each node, corresponding to its frequency of visit by a random walk.

    The random walk restarts with some fixed probability. The restart distribution can be personalized by the user.
    This variant is known as Personalized PageRank.

    Parameters
    ----------
    damping_factor : float
        Probability to continue the random walk.
    solver : str
        Which solver to use: 'spsolve', 'lanczos' (default), 'lsqr' or 'halko'.

    Attributes
    ----------
    score_ : np.ndarray
        PageRank score of each node.

    Example
    -------
    >>> from sknetwork.toy_graphs import rock_paper_scissors
    >>> pagerank = PageRank()
    >>> adjacency = rock_paper_scissors()
    >>> np.round(pagerank.fit(adjacency).score_, 2)
    array([0.33, 0.33, 0.33])

    References
    ----------
    Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web.
    Stanford InfoLab.
    """
    def __init__(self, damping_factor: float = 0.85, solver: str = 'lanczos'):
        if damping_factor < 0 or damping_factor >= 1:
            raise ValueError('Damping factor must be between 0 and 1.')
        else:
            self.damping_factor = damping_factor
        self.solver = solver

        self.score_ = None

    # noinspection PyTypeChecker
    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray],
            personalization: Optional[Union[dict, np.ndarray]] = None, force_biadjacency: bool = False) -> 'PageRank':
        """
        Standard PageRank with restart.

        Parameters
        ----------
        adjacency :
            Adjacency or biadjacency matrix of the graph.
        personalization :
            If ``None``, the uniform distribution is used.
            Otherwise, a non-negative, non-zero vector or a dictionary must be provided.
        force_biadjacency : bool (default= ``False``)
            If ``True``, force the input matrix to be considered as a biadjacency matrix.
        Returns
        -------
        self: :class: 'PageRank'
        """
        adjacency = check_format(adjacency)
        if not is_square(adjacency) or force_biadjacency:
            adjacency = bipartite2undirected(adjacency)
        n: int = adjacency.shape[0]

        if adjacency.nnz:
            rso = RandomSurferOperator(adjacency, self.damping_factor, personalization)

            if self.solver == 'spsolve':
                x = spsolve(sparse.eye(n, format='csr') - rso.a, rso.b)
            elif self.solver == 'lanczos':
                _, x = sparse.linalg.eigs(rso, k=1)
            elif self.solver == 'lsqr':
                x = lsqr(sparse.eye(n, format='csr') - rso.a, rso.b)[0]
            else:
                raise NotImplementedError('Other solvers are not yet available.')

            x = abs(x.flatten().real)
            self.score_ = x / x.sum()

        else:
            self.score_ = np.zeros(n)
        return self
