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

from sknetwork.linalg.operators import CoNeighborsOperator
from sknetwork.linalg.polynome import Polynome
from sknetwork.ranking.base import BaseRanking, BaseBiRanking
from sknetwork.utils.check import check_format
from sknetwork.utils.format import bipartite2undirected


class Katz(BaseRanking):
    """Katz centrality:

    :math:`x_i = \\sum_{k=1}^K\\sum_j \\alpha^k(A^k)_{ij}`.

    * Graphs
    * Digraphs

    Parameters
    ----------
    alpha : float
        Decay parameter for path contributions. Should be less than the spectral radius of the adjacency.
    max_lenght : int
        Maximum lenght of the paths to take into account.

    Examples
    --------
    >>> from sknetwork.data.toy_graphs import house
    >>> adjacency = house()
    >>> scores = Katz().fit_transform(adjacency)
    >>> np.round(scores, 2)
    array([6.5 , 8.25, 5.62, 5.62, 8.25])

    References
    ----------
    Katz, L. (1953). `A new status index derived from sociometric analysis
    <https://link.springer.com/content/pdf/10.1007/BF02289026.pdf>`_. Psychometrika, 18(1), 39-43.
    """
    def __init__(self, alpha: float = 0.5, max_lenght: int = 4):
        super(Katz, self).__init__()
        self.alpha = alpha
        self.max_lenght = max_lenght

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
        coeffs = self.alpha ** np.arange(self.max_lenght + 1)
        coeffs[0] = 0.
        polynome = Polynome(adjacency, coeffs)

        self.scores_ = polynome.dot(np.ones(n))
        return self


class BiKatz(Katz, BaseBiRanking):
    """Katz centrality for bipartite graphs.

    * Bigraphs

    Parameters
    ----------
    alpha : float
        Decay parameter for path contributions. Should be less than the spectral radius of the adjacency.
    max_lenght : int
        Maximum lenght of the paths to take into account.

    Examples
    --------
    >>> from sknetwork.data.toy_graphs import star_wars
    >>> biadjacency = star_wars()
    >>> scores = BiKatz().fit_transform(biadjacency)
    >>> np.round(scores, 2)
    array([6.38, 3.06, 8.81, 5.75])

    References
    ----------
    Katz, L. (1953). `A new status index derived from sociometric analysis
    <https://link.springer.com/content/pdf/10.1007/BF02289026.pdf>`_. Psychometrika, 18(1), 39-43.
    """
    def __init__(self, alpha: float = 0.5, max_lenght: int = 4):
        super(BiKatz, self).__init__(alpha=alpha, max_lenght=max_lenght)

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


class CoKatz(Katz, BaseBiRanking):
    """Katz centrality on the normalized :term:`co-neighbors` graph.

    * Graphs
    * Digraphs
    * Bigraphs

    Parameters
    ----------
    alpha : float
        Decay parameter for path contributions. Should be less than the spectral radius of the adjacency.
    max_lenght : int
        Maximum lenght of the paths to take into account.

    Examples
    --------
    >>> from sknetwork.data.toy_graphs import star_wars
    >>> biadjacency = star_wars()
    >>> scores = CoKatz().fit_transform(biadjacency)
    >>> np.round(scores, 2)
    array([4.68, 2.17, 7.37, 5.2 ])

    References
    ----------
    Katz, L. (1953). `A new status index derived from sociometric analysis
    <https://link.springer.com/content/pdf/10.1007/BF02289026.pdf>`_. Psychometrika, 18(1), 39-43.
    """
    def __init__(self, alpha: float = 0.5, max_lenght: int = 4):
        super(CoKatz, self).__init__(alpha=alpha, max_lenght=max_lenght)

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'CoKatz':
        """Katz centrality.

        Parameters
        ----------
        biadjacency :
            Adjacency or biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`CoKatz`
        """
        katz = Katz(self.alpha, self.max_lenght)
        adjacency = CoNeighborsOperator(biadjacency)
        scores_row = katz.fit_transform(adjacency)

        adjacency = CoNeighborsOperator(biadjacency.T.tocsr())
        scores_col = katz.fit_transform(adjacency)

        self.scores_ = scores_row
        self.scores_row_ = scores_row
        self.scores_col_ = scores_col
        return self
