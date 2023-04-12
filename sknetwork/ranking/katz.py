#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from sknetwork.linalg.polynome import Polynome
from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.check import check_format
from sknetwork.utils.format import get_adjacency


class Katz(BaseRanking):
    """Katz centrality, defined by:

    :math:`\\sum_{k=1}^K\\alpha^k(A^k)^T\\mathbf{1}`

    where :math:`A` is the adjacency matrix, :math:`\\alpha` is the damping factor and :math:`K` is the path length.

    Parameters
    ----------
    damping_factor : float
        Damping factor for path contributions.
    path_length : int
        Maximum length of the paths.

    Attributes
    ----------
    scores_ : np.ndarray
        Score of each node.
    scores_row_: np.ndarray
        Scores of rows, for bipartite graphs.
    scores_col_: np.ndarray
        Scores of columns, for bipartite graphs.

    Examples
    --------
    >>> from sknetwork.data.toy_graphs import house
    >>> adjacency = house()
    >>> katz = Katz()
    >>> scores = katz.fit_predict(adjacency)
    >>> np.round(scores, 2)
    array([6.5 , 8.25, 5.62, 5.62, 8.25])

    References
    ----------
    Katz, L. (1953). `A new status index derived from sociometric analysis
    <https://link.springer.com/content/pdf/10.1007/BF02289026.pdf>`_. Psychometrika, 18(1), 39-43.
    """
    def __init__(self, damping_factor: float = 0.5, path_length: int = 4):
        super(Katz, self).__init__()
        self.damping_factor = damping_factor
        self.path_length = path_length
        self.bipartite = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray, LinearOperator]) -> 'Katz':
        """Katz centrality.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`Katz`
        """
        input_matrix = check_format(input_matrix)
        adjacency, self.bipartite = get_adjacency(input_matrix)
        n = adjacency.shape[0]
        coefs = self.damping_factor ** np.arange(self.path_length + 1)
        coefs[0] = 0.
        polynome = Polynome(adjacency.T.astype(bool).tocsr(), coefs)
        self.scores_ = polynome.dot(np.ones(n))
        if self.bipartite:
            self._split_vars(input_matrix.shape)
        return self
