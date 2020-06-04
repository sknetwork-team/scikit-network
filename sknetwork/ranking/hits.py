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
from sknetwork.ranking.base import BaseBiRanking
from sknetwork.utils.check import check_format


class HITS(BaseBiRanking):
    """Hub and authority scores of each node.
    For bipartite graphs, the hub score is computed on rows and the authority score on columns.

    * Graphs
    * Digraphs
    * Bigraphs

    Parameters
    ----------
    solver : ``'auto'``, ``'halko'``, ``'lanczos'`` or :class:`SVDSolver`
        Which singular value solver to use.

        * ``'auto'`` call the auto_solver function.
        * ``'halko'``: randomized method, fast but less accurate than ``'lanczos'`` for ill-conditioned matrices.
        * ``'lanczos'``: power-iteration based method.
        * :class:`SVDSolver`: custom solver.

    **kwargs :
        See :ref:`sknetwork.linalg.svd_solver.LanczosSVD <lanczossvd>`
        or :ref:`sknetwork.linalg.svd_solver.HalkoSVD <halkosvd>`.

    Attributes
    ----------
    scores_ : np.ndarray
        Hub score of each row.
    scores_row_ : np.ndarray
        Hub score of each row (copy of **scores_row_**).
    scores_col_ : np.ndarray
        Authority score of each column.

    Example
    -------
    >>> from sknetwork.ranking import HITS
    >>> from sknetwork.data import star_wars
    >>> hits = HITS()
    >>> biadjacency = star_wars()
    >>> scores = hits.fit_transform(biadjacency)
    >>> np.round(scores, 2)
    array([0.5 , 0.23, 0.69, 0.46])

    References
    ----------
    Kleinberg, J. M. (1999). Authoritative sources in a hyperlinked environment.
    Journal of the ACM (JACM), 46(5), 604-632.
    """
    def __init__(self, solver: Union[str, SVDSolver] = 'auto', **kwargs):
        super(HITS, self).__init__()

        if solver == 'halko':
            self.solver: SVDSolver = HalkoSVD(**kwargs)
        elif solver == 'lanczos':
            self.solver: SVDSolver = LanczosSVD(**kwargs)
        else:
            self.solver = solver

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'HITS':
        """Compute HITS algorithm with a spectral method.

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
            else:  # pragma: no cover
                self.solver: SVDSolver = HalkoSVD()

        self.solver.fit(adjacency, 1)
        hubs: np.ndarray = self.solver.singular_vectors_left_.reshape(-1)
        authorities: np.ndarray = self.solver.singular_vectors_right_.reshape(-1)

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

        self.scores_row_ = hubs
        self.scores_col_ = authorities
        self.scores_ = hubs

        return self
