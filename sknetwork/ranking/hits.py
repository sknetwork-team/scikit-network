#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 07 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.linalg import SVDSolver, HalkoSVD, LanczosSVD, auto_solver
from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.checks import check_format


class HITS(BaseRanking):
    """
    Compute the hub and authority weights of each node.
    For bipartite graphs, the hub score is computed on one part and the authority score on the other one.

    Parameters
    ----------
    mode:
        Either ``'hubs'`` or ``'authorities'``. Which of the weights to store in the ``.scores_`` attribute.
        The other one is stored in ``.col_scores_``.
    solver: ``'auto'``, ``'halko'``, ``'lanczos'`` or :class:`SVDSolver`
        Which singular value solver to use.

        * ``'auto'`` call the auto_solver function.
        * ``'halko'``: randomized method, fast but less accurate than ``'lanczos'`` for ill-conditioned matrices.
        * ``'lanczos'``: power-iteration based method.
        * :class:`SVDSolver`: custom solver.

    Attributes
    ----------
    scores_ : np.ndarray
        Hub or authority score of each node, depending on the value of **mode**.
    col_scores_ : np.ndarray
        Hub or authority score of each node, depending on the value of **mode**.

    Example
    -------
    >>> from sknetwork.data import star_wars_villains
    >>> hits = HITS()
    >>> biadjacency: sparse.csr_matrix = star_wars_villains()
    >>> np.round(hits.fit(biadjacency).scores_, 2)
    array([0.5 , 0.23, 0.69, 0.46])
    >>> np.round(hits.col_scores_, 2)
    array([0.58, 0.47, 0.67])

    References
    ----------
    Kleinberg, J. M. (1999). Authoritative sources in a hyperlinked environment.
    Journal of the ACM (JACM), 46(5), 604-632.

    """

    def __init__(self, mode: str = 'hubs', solver: Union[str, SVDSolver] = 'auto'):
        super(HITS, self).__init__()

        self.mode = mode
        if solver == 'halko':
            self.solver: SVDSolver = HalkoSVD()
        elif solver == 'lanczos':
            self.solver: SVDSolver = LanczosSVD()
        else:
            self.solver = solver

        self.col_scores_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'HITS':
        """
        Compute HITS algorithm with a spectral method.

        Parameters
        ----------
        adjacency :
            Adjacency or biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`HITS`
        """
        adjacency = check_format(adjacency)

        if self.solver == 'auto':
            solver = auto_solver(adjacency.nnz)
            if solver == 'lanczos':
                self.solver: SVDSolver = LanczosSVD()
            else:
                self.solver: SVDSolver = HalkoSVD()

        self.solver.fit(adjacency, 1)
        hubs: np.ndarray = self.solver.left_singular_vectors_.reshape(-1)
        authorities: np.ndarray = self.solver.right_singular_vectors_.reshape(-1)

        h_pos, h_neg = (hubs > 0).sum(), (hubs < 0).sum()
        a_pos, a_neg = (authorities > 0).sum(), (authorities < 0).sum()

        if h_pos > h_neg:
            hubs = np.clip(hubs, a_min=0., a_max=None)
        else:
            hubs = np.clip(-hubs, a_min=0., a_max=None)

        if a_pos > a_neg:
            authorities = np.clip(authorities, a_min=0., a_max=None)
        else:
            authorities = np.clip(-authorities, a_min=0., a_max=None)

        if self.mode == 'hubs':
            self.scores_ = hubs
            self.col_scores_ = authorities
        elif self.mode == 'authorities':
            self.scores_ = authorities
            self.col_scores_ = hubs
        else:
            raise ValueError('Mode should be "hubs" or "authorities".')

        return self
