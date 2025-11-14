#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
@author: Thomas Bonald <thomas.bonald@telecom-paris.fr>
"""
from typing import Union, Optional, Tuple

import numpy as np
from scipy import sparse

from sknetwork.linalg.normalizer import normalize
from sknetwork.regression.base import BaseRegressor
from sknetwork.utils import get_adjacency_values, get_degrees


def init_temperatures(seeds: np.ndarray, init: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Init temperatures."""
    n = len(seeds)
    border = (seeds >= 0)
    if init is None:
        temperatures = seeds[border].mean() * np.ones(n)
    else:
        temperatures = init * np.ones(n)
    temperatures[border] = seeds[border]
    return temperatures, border


class Diffusion(BaseRegressor):
    """Regression by diffusion along the edges, given the temperatures of some seed nodes (heat equation).

    The row vector of tempreatures :math:`T` evolves like:

    :math:`T \\gets (1-\\alpha) T + \\alpha  PT`

    where :math:`\\alpha` is the damping factor and :math:`P` is the transition matrix of the random walk in the graph.

    All values are updated, including those of seed nodes (free diffusion).
    See ``Dirichlet`` for diffusion with boundary constraints.

    Parameters
    ----------
    n_iter : int
        Number of iterations of the diffusion (must be positive).
    damping_factor : float
        Damping factor.

    Attributes
    ----------
    values\_ : np.ndarray
        Value of each node (= temperature).

    Example
    -------
    >>> from sknetwork.data import house
    >>> diffusion = Diffusion(n_iter=1)
    >>> adjacency = house()
    >>> values = {0: 1, 2: 0}
    >>> values_pred = diffusion.fit_predict(adjacency, values)
    >>> np.round(values_pred, 1)
    array([0.8, 0.5, 0.2, 0.4, 0.6])

    References
    ----------
    Chung, F. (2007). The heat kernel as the pagerank of a graph. Proceedings of the National Academy of Sciences.
    """
    def __init__(self, n_iter: int = 3, damping_factor: float = 0.5):
        super(Diffusion, self).__init__()

        if n_iter <= 0:
            raise ValueError('The number of iterations must be positive.')
        else:
            self.n_iter = n_iter
        self.damping_factor = damping_factor
        self.bipartite = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray],
            values: Optional[Union[dict, list, np.ndarray]] = None,
            values_row: Optional[Union[dict, list, np.ndarray]] = None,
            values_col: Optional[Union[dict, list, np.ndarray]] = None, init: Optional[float] = None,
            force_bipartite: bool = False) -> 'Diffusion':
        """Compute the diffusion (temperatures at equilibrium).

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        values :
            Temperatures of nodes in initial state (dictionary or vector). Negative temperatures ignored.
        values_row, values_col :
            Temperatures of rows and columns for bipartite graphs. Negative temperatures ignored.
        init :
            Temperature of nodes in initial state.
            If ``None``, use the average temperature of seed nodes (default).
        force_bipartite :
            If ``True``, consider the input matrix as a biadjacency matrix (default = ``False``).

        Returns
        -------
        self: :class:`Diffusion`
        """
        adjacency, values, self.bipartite = get_adjacency_values(input_matrix, force_bipartite=force_bipartite,
                                                                 values=values,
                                                                 values_row=values_row,
                                                                 values_col=values_col)
        values, _ = init_temperatures(values, init)
        diffusion = normalize(adjacency.T.tocsr())
        degrees = get_degrees(diffusion)
        diag = sparse.diags((degrees == 0).astype(int)).tocsr()
        diffusion += diag

        diffusion = (1 - self.damping_factor) * sparse.identity(len(degrees)).tocsr() + self.damping_factor * diffusion

        for i in range(self.n_iter):
            values = diffusion.dot(values)

        self.values_ = values
        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self


class Dirichlet(BaseRegressor):
    """Regression by the Dirichlet problem (heat diffusion with boundary constraints).

    The temperatures of some seed nodes are fixed. The temperatures of other nodes are computed.

    Parameters
    ----------
    n_iter : int
        Number of iterations of the diffusion (must be positive).

    Attributes
    ----------
    values\_ : np.ndarray
        Value of each node (= temperature).

    Example
    -------
    >>> from sknetwork.regression import Dirichlet
    >>> from sknetwork.data import house
    >>> dirichlet = Dirichlet()
    >>> adjacency = house()
    >>> values = {0: 1, 2: 0}
    >>> values_pred = dirichlet.fit_predict(adjacency, values)
    >>> np.round(values_pred, 2)
    array([1.  , 0.54, 0.  , 0.31, 0.62])

    References
    ----------
    Chung, F. (2007). The heat kernel as the pagerank of a graph. Proceedings of the National Academy of Sciences.
    """
    def __init__(self, n_iter: int = 10):
        super(Dirichlet, self).__init__()

        if n_iter <= 0:
            raise ValueError('The number of iterations must be positive.')
        else:
            self.n_iter = n_iter
        self.bipartite = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray],
            values: Optional[Union[dict, list, np.ndarray]] = None,
            values_row: Optional[Union[dict, list, np.ndarray]] = None,
            values_col: Optional[Union[dict, list, np.ndarray]] = None, init: Optional[float] = None,
            force_bipartite: bool = False) -> 'Dirichlet':
        """Compute the solution to the Dirichlet problem (temperatures at equilibrium).

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        values :
            Temperatures of nodes (dictionary or vector). Negative temperatures ignored.
        values_row, values_col :
            Temperatures of rows and columns for bipartite graphs. Negative temperatures ignored.
        init :
            Temperature of nodes in initial state.
            If ``None``, use the average temperature of seed nodes (default).
        force_bipartite :
            If ``True``, consider the input matrix as a biadjacency matrix (default = ``False``).

        Returns
        -------
        self: :class:`Dirichlet`
        """
        adjacency, values, self.bipartite = get_adjacency_values(input_matrix, force_bipartite=force_bipartite,
                                                                 values=values,
                                                                 values_row=values_row,
                                                                 values_col=values_col)
        temperatures, border = init_temperatures(values, init)
        values = temperatures.copy()
        diffusion = normalize(adjacency)
        for i in range(self.n_iter):
            values = diffusion.dot(values)
            values[border] = temperatures[border]

        self.values_ = values
        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self
