#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from sknetwork.linalg.polynome import Polynome
from sknetwork.ranking.base import BaseRanking, BaseBiRanking
from sknetwork.utils.check import check_format
from sknetwork.utils.format import bipartite2undirected


class Katz(BaseRanking):
    """Katz centrality:

    :math:`\\sum_{k=1}^K\\alpha^k(A^k)^T\\mathbf{1}`.

    * Graphs
    * Digraphs

    Parameters
    ----------
    damping_factor : float
        Decay parameter for path contributions.
    path_length : int
        Maximum length of the paths to take into account.

    Examples
    --------
    >>> from sknetwork.data.toy_graphs import house
    >>> adjacency = house()
    >>> katz = Katz()
    >>> scores = katz.fit_transform(adjacency)
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

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray, LinearOperator]) -> 'Katz':
        """Katz centrality.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`Katz`
        """
        if not isinstance(adjacency, LinearOperator):
            adjacency = check_format(adjacency)
        n = adjacency.shape[0]
        coeffs = self.damping_factor ** np.arange(self.path_length + 1)
        coeffs[0] = 0.
        polynome = Polynome(adjacency.T.astype(bool).tocsr(), coeffs)

        self.scores_ = polynome.dot(np.ones(n))
        return self


class BiKatz(Katz, BaseBiRanking):
    """Katz centrality for bipartite graphs.

    * Bigraphs

    Parameters
    ----------
    damping_factor : float
        Decay parameter for path contributions.
    path_length : int
        Maximum length of the paths to take into account.

    Examples
    --------
    >>> from sknetwork.data.toy_graphs import star_wars
    >>> biadjacency = star_wars()
    >>> bikatz = BiKatz()
    >>> scores = bikatz.fit_transform(biadjacency)
    >>> np.round(scores, 2)
    array([6.38, 3.06, 8.81, 5.75])

    References
    ----------
    Katz, L. (1953). `A new status index derived from sociometric analysis
    <https://link.springer.com/content/pdf/10.1007/BF02289026.pdf>`_. Psychometrika, 18(1), 39-43.
    """
    def __init__(self, damping_factor: float = 0.5, path_length: int = 4):
        super(BiKatz, self).__init__(damping_factor=damping_factor, path_length=path_length)

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiKatz':
        """Katz centrality.

        Parameters
        ----------
        biadjacency :
            Biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiKatz`
        """
        biadjacency = check_format(biadjacency)
        n_row, n_col = biadjacency.shape
        adjacency = bipartite2undirected(biadjacency)

        Katz.fit(self, adjacency)
        self._split_vars(n_row)
        return self
