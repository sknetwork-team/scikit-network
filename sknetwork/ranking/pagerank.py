#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in May 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.linalg.ppr_solver import get_pagerank
from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.check import check_damping_factor
from sknetwork.utils.format import get_adjacency_values


class PageRank(BaseRanking):
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
    scores\_ : np.ndarray
        PageRank score of each node.

    Example
    -------
    >>> from sknetwork.ranking import PageRank
    >>> from sknetwork.data import house
    >>> pagerank = PageRank()
    >>> adjacency = house()
    >>> weights = {0: 1}
    >>> scores = pagerank.fit_predict(adjacency, weights)
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

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray],
            weights: Optional[Union[dict, np.ndarray]] = None, weights_row: Optional[Union[dict, np.ndarray]] = None,
            weights_col: Optional[Union[dict, np.ndarray]] = None, force_bipartite: bool = False) -> 'PageRank':
        """Compute the pagerank of each node.

        Parameters
        ----------
        input_matrix : sparse.csr_matrix, np.ndarray
            Adjacency matrix or biadjacency matrix of the graph.
        weights : np.ndarray, dict
            Weights of the restart distribution for Personalized PageRank.
            If ``None``, the uniform distribution is used (no personalization, default).
        weights_row : np.ndarray, dict
            Weights on rows of the restart distribution for Personalized PageRank.
            Used for bipartite graphs.
            If both weights_row and weights_col are ``None`` (default), the uniform distribution on rows is used.
        weights_col : np.ndarray, dict
            Weights on columns of the restart distribution for Personalized PageRank.
            Used for bipartite graphs.
        force_bipartite : bool
            If ``True``, consider the input matrix as the biadjacency matrix of a bipartite graph.
        Returns
        -------
        self: :class:`PageRank`
        """
        adjacency, values, self.bipartite = get_adjacency_values(input_matrix, force_bipartite=force_bipartite,
                                                                 values=weights,
                                                                 values_row=weights_row,
                                                                 values_col=weights_col,
                                                                 default_value=0,
                                                                 which='probs')
        self.scores_ = get_pagerank(adjacency, values, damping_factor=self.damping_factor, n_iter=self.n_iter,
                                    solver=self.solver, tol=self.tol)
        if self.bipartite:
            self._split_vars(input_matrix.shape)
        return self
