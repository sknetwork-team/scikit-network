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
from scipy.sparse.linalg import LinearOperator

from sknetwork.linalg.operators import CoNeighborsOperator
from sknetwork.linalg.ppr_solver import get_pagerank
from sknetwork.ranking.base import BaseRanking, BaseBiRanking
from sknetwork.utils.format import bipartite2undirected
from sknetwork.utils.check import check_format, check_square
from sknetwork.utils.seeds import seeds2probs, stack_seeds
from sknetwork.utils.verbose import VerboseMixin


class PageRank(BaseRanking, VerboseMixin):
    """PageRank of each node, corresponding to its frequency of visit by a random walk.

    The random walk restarts with some fixed probability. The restart distribution can be personalized by the user.
    This variant is known as Personalized PageRank.

    * Graphs
    * Digraphs

    Parameters
    ----------
    damping_factor : float
        Probability to continue the random walk.
    solver : str
        * `piteration`, use power iteration for a given number of iterations.
        * `diteration`, use asynchronous parallel diffusion for a given number of iterations.
        * `lanczos`, use eigensolver with a given tolerance.
        * `bicgstab`, use Biconjugate Gradient Stabilized method for a given tolerance.
    n_iter : int
        Number of iterations for some solvers.
    tol : float
        Tolerance for the convergence of some solvers.

    Attributes
    ----------
    scores_ : np.ndarray
        PageRank score of each node.

    Example
    -------
    >>> from sknetwork.ranking import PageRank
    >>> from sknetwork.data import house
    >>> pagerank = PageRank()
    >>> adjacency = house()
    >>> seeds = {0: 1}
    >>> scores = pagerank.fit_transform(adjacency, seeds)
    >>> np.round(scores, 2)
    array([0.29, 0.24, 0.12, 0.12, 0.24])

    References
    ----------
    Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web.
    Stanford InfoLab.
    """
    def __init__(self, damping_factor: float = 0.85, solver: str = 'piteration', n_iter: int = 10, tol: float = 0):
        super(PageRank, self).__init__()

        if damping_factor < 0 or damping_factor >= 1:
            raise ValueError('Damping factor must be between 0 and 1.')
        else:
            self.damping_factor = damping_factor
        self.solver = solver
        self.n_iter = n_iter
        self.tol = tol

    # noinspection PyTypeChecker
    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray, LinearOperator],
            seeds: Optional[Union[dict, np.ndarray]] = None) -> 'PageRank':
        """Fit algorithm to data.

        Parameters
        ----------
        adjacency :
            Adjacency matrix.
        seeds :
            If ``None``, the uniform distribution is used.
            Otherwise, a non-negative, non-zero vector or a dictionary must be provided.

        Returns
        -------
        self: :class:`PageRank`
        """
        if not isinstance(adjacency, LinearOperator):
            adjacency = check_format(adjacency)
        check_square(adjacency)
        seeds = seeds2probs(adjacency.shape[0], seeds)
        self.scores_ = get_pagerank(adjacency, seeds, damping_factor=self.damping_factor, n_iter=self.n_iter,
                                    solver=self.solver, tol=self.tol)

        return self


class BiPageRank(PageRank, BaseBiRanking):
    """Compute the PageRank of each node through a random walk in the bipartite graph.

    * Bigraphs

    Parameters
    ----------
    damping_factor : float
        Probability to continue the random walk.
    solver : str
        * `piteration`, use power iteration for a given number of iterations.
        * `diteration`, use asynchronous parallel diffusion for a given number of iterations.
        * `lanczos`, use eigensolver for a given tolerance.
        * `bicgstab`, use Biconjugate Gradient Stabilized method for a given tolerance.
    n_iter : int
        Number of iterations for some solvers.
    tol : float
        Tolerance for the convergence of some solvers.

    Attributes
    ----------
    scores_ : np.ndarray
        PageRank score of each row.
    scores_row_ : np.ndarray
        PageRank score of each row (copy of **scores_**).
    scores_col_ : np.ndarray
        PageRank score of each column.

    Example
    -------
    >>> from sknetwork.ranking import BiPageRank
    >>> from sknetwork.data import star_wars
    >>> bipagerank = BiPageRank()
    >>> biadjacency = star_wars()
    >>> seeds = {0: 1}
    >>> scores = bipagerank.fit_transform(biadjacency, seeds)
    >>> np.round(scores, 2)
    array([0.45, 0.11, 0.28, 0.17])
    """
    def __init__(self, damping_factor: float = 0.85, solver: str = 'piteration', n_iter: int = 10, tol: float = 0):
        super(BiPageRank, self).__init__(damping_factor, solver, n_iter, tol)

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray],
            seeds_row: Optional[Union[dict, np.ndarray]] = None, seeds_col: Optional[Union[dict, np.ndarray]] = None) \
            -> 'BiPageRank':
        """Fit algorithm to data.

        Parameters
        ----------
        biadjacency :
            Biadjacency matrix.
        seeds_row :
            Seed rows, as a dict or a vector.
        seeds_col :
            Seed columns, as a dict or a vector.
            If both seeds_row and seeds_col are ``None``, the uniform distribution is used.

        Returns
        -------
        self: :class:`BiPageRank`
        """
        biadjacency = check_format(biadjacency)
        n_row, n_col = biadjacency.shape
        adjacency = bipartite2undirected(biadjacency)
        seeds = stack_seeds(n_row, n_col, seeds_row, seeds_col)

        PageRank.fit(self, adjacency, seeds)
        self._split_vars(n_row)

        self.scores_row_ /= self.scores_row_.sum()
        self.scores_col_ /= self.scores_col_.sum()
        self.scores_ = self.scores_row_

        return self


class CoPageRank(BiPageRank):
    """Compute the PageRank of each node through a two-hops random walk in the bipartite graph.

    * Graphs
    * Digraphs
    * Bigraphs

    Parameters
    ----------
    damping_factor : float
        Probability to continue the random walk.
    solver : str
        * `piteration`, use power iteration for a given number of iterations.
        * `lanczos`, use eigensolver for a given tolerance.
        * `bicgstab`, use Biconjugate Gradient Stabilized method for a given tolerance.
    n_iter : int
        Number of iterations for some solvers.
    tol : float
        Tolerance for the convergence of some solvers.

    Attributes
    ----------
    scores_ : np.ndarray
        PageRank score of each row.
    scores_row_ : np.ndarray
        PageRank score of each row (copy of **scores_**).
    scores_col_ : np.ndarray
        PageRank score of each column.

    Example
    -------
    >>> from sknetwork.ranking import CoPageRank
    >>> from sknetwork.data import star_wars
    >>> copagerank = CoPageRank()
    >>> biadjacency = star_wars()
    >>> seeds = {0: 1}
    >>> scores = copagerank.fit_transform(biadjacency, seeds)
    >>> np.round(scores, 2)
    array([0.38, 0.12, 0.31, 0.2 ])
    """
    def __init__(self, damping_factor: float = 0.85, solver: str = 'piteration', n_iter: int = 10, tol: float = 0):
        super(CoPageRank, self).__init__(damping_factor, solver, n_iter, tol)

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray],
            seeds_row: Optional[Union[dict, np.ndarray]] = None,
            seeds_col: Optional[Union[dict, np.ndarray]] = None) -> 'CoPageRank':
        """Fit algorithm to data.

        Parameters
        ----------
        biadjacency :
            Biadjacency matrix.
        seeds_row :
            Seed rows, as a dict or a vector.
        seeds_col :
            Seed columns, as a dict or a vector.
            If both seeds_row and seeds_col are ``None``, the uniform distribution is used.

        Returns
        -------
        self: :class:`CoPageRank`
        """
        biadjacency = check_format(biadjacency)
        n_row, n_col = biadjacency.shape

        operator = CoNeighborsOperator(biadjacency, True)
        seeds_row = seeds2probs(n_row, seeds_row)
        self.scores_row_ = get_pagerank(operator, seeds_row, damping_factor=self.damping_factor, solver=self.solver,
                                        n_iter=self.n_iter, tol=self.tol)

        operator = CoNeighborsOperator(biadjacency.T.tocsr(), True)
        seeds_col = seeds2probs(n_col, seeds_col)
        self.scores_col_ = get_pagerank(operator, seeds_col, damping_factor=self.damping_factor, solver=self.solver,
                                        n_iter=self.n_iter, tol=self.tol)

        self.scores_ = self.scores_row_

        return self
