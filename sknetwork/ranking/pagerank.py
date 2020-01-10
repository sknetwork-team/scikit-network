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
from scipy.sparse.linalg import eigs, LinearOperator, lsqr, bicgstab

from sknetwork.basics.rand_walk import transition_matrix
from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.adjacency_formats import bipartite2undirected
from sknetwork.utils.checks import check_format, has_nonnegative_entries, is_square
from sknetwork.utils.verbose import VerboseMixin


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
            raise ValueError('Personalization must be None or a non-negative, non-null vector '
                             'or a dictionary with positive values.')
    return restart_prob


class RandomSurferOperator(LinearOperator, VerboseMixin):
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
    fb_mode :
        Forward-Backward mode. If ``True``, the random surfer performs two consecutive jumps, the first one follows the
        direction of the edges, while the second one goes in the opposite direction.

    Attributes
    ----------
    a : sparse.csr_matrix
        Scaled transposed transition matrix.
    b : np.ndarray
        Scaled restart probability vector.

    """
    def __init__(self, adjacency: sparse.csr_matrix, damping_factor: float = 0.85, personalization=None,
                 fb_mode: bool = False, verbose: bool = False):
        VerboseMixin.__init__(self, verbose)

        n1, n2 = adjacency.shape
        restart_prob: np.ndarray = restart_probability(n1, personalization)

        if fb_mode:
            restart_prob = np.hstack((restart_prob, np.zeros(n2)))
            adjacency = bipartite2undirected(adjacency)

        LinearOperator.__init__(self, shape=adjacency.shape, dtype=float)
        n = adjacency.shape[0]
        out_degrees = adjacency.dot(np.ones(n))

        damping_matrix = damping_factor * sparse.eye(n, format='csr')
        if fb_mode:
            damping_matrix.data[n1:] = 1

        self.a = (damping_matrix.dot(transition_matrix(adjacency))).T.tocsr()
        self.b = (np.ones(n) - damping_factor * out_degrees.astype(bool)) * restart_prob

    def _matvec(self, x):
        return self.a.dot(x) + self.b * x.sum()

    # noinspection PyTypeChecker
    def solve(self, solver: str = 'lanczos', n_iter: int = 10):
        """Pagerank vector for a given adjacency and personalization.

        Parameters
        ----------
        solver: str
            Which method to use to solve the Pagerank problem. Can be 'lanczos', 'lsqr' or 'bicgstab'.
        n_iter : int
        If ``solver`` is not one of the standard values, the pagerank is approximated by emulating the random walk for
        ``n_iter`` iterations.

        Returns
        -------
        score: np.ndarray
            Pagerank of the rows.

        """

        n: int = self.a.shape[0]

        if solver == 'bicgstab':
            x, info = bicgstab(sparse.eye(n, format='csr') - self.a, self.b, atol=0.)
            self.scipy_solver_info(info)
        elif solver == 'lanczos':
            _, x = sparse.linalg.eigs(self, k=1)
        elif solver == 'lsqr':
            x = lsqr(sparse.eye(n, format='csr') - self.a, self.b)[0]
        else:
            x = self.b
            for i in range(n_iter):
                x = self.dot(x)
                x /= x.sum()

        x = abs(x.flatten().real)
        return x / x.sum()


class PageRank(BaseRanking, VerboseMixin):
    """
    Compute the PageRank of each node, corresponding to its frequency of visit by a random walk.

    The random walk restarts with some fixed probability. The restart distribution can be personalized by the user.
    This variant is known as Personalized PageRank.

    Parameters
    ----------
    damping_factor : float
        Probability to continue the random walk.
    solver : str
        Which solver to use: 'spsolve', 'lanczos' (default), 'lsqr' or 'halko'.
        Otherwise, the random walk is emulated for a certain number of iterations.
    n_iter : int
        If ``solver`` is not one of the standard values, the pagerank is approximated by emulating the random walk for
        ``n_iter`` iterations.

    Attributes
    ----------
    scores_ : np.ndarray
        PageRank score of each node.

    Example
    -------
    >>> from sknetwork.data import rock_paper_scissors
    >>> pagerank = PageRank()
    >>> adjacency = rock_paper_scissors()
    >>> np.round(pagerank.fit_transform(adjacency), 2)
    array([0.33, 0.33, 0.33])

    References
    ----------
    Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web.
    Stanford InfoLab.
    """
    def __init__(self, damping_factor: float = 0.85, solver: Union[str, None] = 'lanczos', n_iter: int = 10):
        super(PageRank, self).__init__()

        if damping_factor < 0 or damping_factor >= 1:
            raise ValueError('Damping factor must be between 0 and 1.')
        else:
            self.damping_factor = damping_factor
        self.solver = solver
        self.n_iter = n_iter

    # noinspection PyTypeChecker
    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray],
            personalization: Optional[Union[dict, np.ndarray]] = None) -> 'PageRank':
        """
        Standard PageRank with restart.

        Parameters
        ----------
        adjacency :
            Adjacency matrix.
        personalization :
            If ``None``, the uniform distribution is used.
            Otherwise, a non-negative, non-zero vector or a dictionary must be provided.

        Returns
        -------
        self: :class:`PageRank`
        """

        adjacency = check_format(adjacency)
        if not is_square(adjacency):
            raise ValueError("The adjacency is not square. See BiPageRank.")

        rso = RandomSurferOperator(adjacency, self.damping_factor, personalization, False)
        self.scores_ = rso.solve(self.solver, self.n_iter)

        return self


class BiPageRank(PageRank):
    """
    Compute the PageRank of each node through a two-hop random walk in the bipartite graph.
    The random walk restarts with some fixed probability. The restart distribution can be personalized.

    Parameters
    ----------
    damping_factor : float
        Probability to continue the random walk.
    solver : str
        Which solver to use: 'bicgstab', 'lanczos', 'lsqr'.

    Attributes
    ----------
    row_scores_ : np.ndarray
        PageRank score of each row.
    col_scores_ : np.ndarray
        PageRank score of each col.
    scores_ : np.ndarray
        PageRank score of each node (concatenation of row scores and col scores).

    Example
    -------
    >>> from sknetwork.data import star_wars_villains
    >>> bipagerank = BiPageRank()
    >>> biadjacency: sparse.csr_matrix = star_wars_villains()
    >>> biadjacency.shape
    (4, 3)
    >>> len(bipagerank.fit_transform(biadjacency))
    7
    """
    def __init__(self, damping_factor: float = 0.85, solver: str = 'lanczos', n_iter: int = 10):
        PageRank.__init__(self, damping_factor, solver, n_iter=n_iter)

        self.row_scores_ = None
        self.col_scores_ = None

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray],
            personalization: Optional[Union[dict, np.ndarray]] = None) -> 'BiPageRank':
        """
        Two hops PageRank with restart.

        Parameters
        ----------
        biadjacency :
            Adjacency matrix.
        personalization :
            If ``None``, the uniform distribution is used.
            Otherwise, a non-negative, non-zero vector or a dictionary must be provided.

        Returns
        -------
        self: :class:`BiPageRank`
        """

        rso = RandomSurferOperator(biadjacency, self.damping_factor, personalization, True)
        self.row_scores_ = rso.solve(self.solver, self.n_iter)[:biadjacency.shape[0]]
        self.col_scores_ = transition_matrix(biadjacency.T).dot(self.row_scores_)
        self.col_scores_ /= self.col_scores_.sum()
        self.scores_ = np.concatenate((self.row_scores_, self.col_scores_))

        return self
