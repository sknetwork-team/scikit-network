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

from sknetwork.linalg.ppr_solver import get_pagerank
from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.check import check_damping_factor
from sknetwork.utils.format import get_adjacency_seeds
from sknetwork.utils.verbose import VerboseMixin


class PageRank(BaseRanking, VerboseMixin):
    """PageRank of each node, corresponding to its frequency of visit by a random walk.

    The random walk restarts with some fixed probability. The restart distribution can be personalized by the user.
    This variant is known as Personalized PageRank.

    Parameters
    ----------
    damping_factor : float
        Probability to continue the random walk.
    solver : str
        * ``'piteration'``, use power iteration for a given number of iterations.
        * ``'diteration'``, use asynchronous parallel diffusion for a given number of iterations.
        * ``'lanczos'``, use eigensolver with a given tolerance.
        * ``'bicgstab'``, use Biconjugate Gradient Stabilized method for a given tolerance.
        * ``'RH'``, use a Ruffini-Horner polynomial evaluation.
        * ``'push'``, use push-based algorithm for a given tolerance
    n_iter : int
        Number of iterations for some solvers.
    tol : float
        Tolerance for the convergence of some solvers.

    Attributes
    ----------
    scores_ : np.ndarray
        PageRank score of each node.
    scores_row_: np.ndarray
        Scores of rows, for bipartite graphs.
    scores_col_: np.ndarray
        Scores of columns, for bipartite graphs.

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
    def __init__(self, damping_factor: float = 0.85, solver: str = 'piteration', n_iter: int = 10, tol: float = 1e-6):
        super(PageRank, self).__init__()
        check_damping_factor(damping_factor)
        self.damping_factor = damping_factor
        self.solver = solver
        self.n_iter = n_iter
        self.tol = tol
        self.bipartite = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray, LinearOperator],
            seeds: Optional[Union[dict, np.ndarray]] = None, seeds_row: Optional[Union[dict, np.ndarray]] = None,
            seeds_col: Optional[Union[dict, np.ndarray]] = None, force_bipartite: bool = False) -> 'PageRank':
        """Fit algorithm to data.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        seeds :
            Parameter to be used for Personalized PageRank.
            Restart distribution as a vector or a dict (node: weight).
            If ``None``, the uniform distribution is used (no personalization, default).
        seeds_row, seeds_col :
            Parameter to be used for Personalized PageRank on bipartite graphs.
            Restart distribution as vectors or dicts on rows, columns (node: weight).
            If both seeds_row and seeds_col are ``None`` (default), the uniform distribution on rows is used.
        force_bipartite :
            If ``True``, consider the input matrix as the biadjacency matrix of a bipartite graph.
        Returns
        -------
        self: :class:`PageRank`
        """
        adjacency, seeds, self.bipartite = get_adjacency_seeds(input_matrix, force_bipartite=force_bipartite,
                                                               seeds=seeds, seeds_row=seeds_row,
                                                               seeds_col=seeds_col, default_value=0, which='probs')
        self.scores_ = get_pagerank(adjacency, seeds, damping_factor=self.damping_factor, n_iter=self.n_iter,
                                    solver=self.solver, tol=self.tol)
        if self.bipartite:
            self._split_vars(input_matrix.shape)
        return self
