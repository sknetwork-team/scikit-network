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

from sknetwork.linalg.normalization import normalize
from sknetwork.regression.base import BaseRegressor
from sknetwork.utils.format import get_adjacency_seeds


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

    All values are updated, including those of seed nodes (free diffusion).
    See ``Dirichlet`` for diffusion with boundary constraints.

    Parameters
    ----------
    n_iter : int
        Number of iterations of the diffusion (must be positive).

    Attributes
    ----------
    values_ : np.ndarray
        Value of each node (= temperature).
    values_row_: np.ndarray
        Values of rows, for bipartite graphs.
    values_col_: np.ndarray
        Values of columns, for bipartite graphs.
    Example
    -------
    >>> from sknetwork.data import house
    >>> diffusion = Diffusion(n_iter=2)
    >>> adjacency = house()
    >>> seeds = {0: 1, 2: 0}
    >>> values = diffusion.fit_predict(adjacency, seeds)
    >>> np.round(values, 2)
    array([0.58, 0.56, 0.38, 0.58, 0.42])

    References
    ----------
    Chung, F. (2007). The heat kernel as the pagerank of a graph. Proceedings of the National Academy of Sciences.
    """
    def __init__(self, n_iter: int = 3):
        super(Diffusion, self).__init__()

        if n_iter <= 0:
            raise ValueError('The number of iterations must be positive.')
        else:
            self.n_iter = n_iter
        self.bipartite = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray],
            seeds: Optional[Union[dict, np.ndarray]] = None, seeds_row: Optional[Union[dict, np.ndarray]] = None,
            seeds_col: Optional[Union[dict, np.ndarray]] = None, init: Optional[float] = None,
            force_bipartite: bool = False) -> 'Diffusion':
        """Compute the diffusion (temperatures at equilibrium).

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        seeds :
            Temperatures of seed nodes in initial state (dictionary or vector). Negative temperatures ignored.
        seeds_row, seeds_col :
            Temperatures of rows and columns for bipartite graphs. Negative temperatures ignored.
        init :
            Temperature of non-seed nodes in initial state.
            If ``None``, use the average temperature of seed nodes (default).
        force_bipartite :
            If ``True``, consider the input matrix as a biadjacency matrix (default = ``False``).

        Returns
        -------
        self: :class:`Diffusion`
        """
        adjacency, seeds, self.bipartite = get_adjacency_seeds(input_matrix, force_bipartite=force_bipartite,
                                                               seeds=seeds, seeds_row=seeds_row, seeds_col=seeds_col)
        values, _ = init_temperatures(seeds, init)
        diffusion = normalize(adjacency)
        for i in range(self.n_iter):
            values = diffusion.dot(values)

        self.values_ = values
        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self


class Dirichlet(BaseRegressor):
    """Regression by the Dirichlet problem, given the temperature of some seed nodes
     (heat diffusion with boundary constraints).

     Only values of non-seed nodes are updated. The temperatures of seed nodes are fixed.

    Parameters
    ----------
    n_iter : int
        Number of iterations of the diffusion (must be positive).

    Attributes
    ----------
    values_ : np.ndarray
        Value of each node (= temperature).
    values_row_: np.ndarray
        Values of rows, for bipartite graphs.
    values_col_: np.ndarray
        Values of columns, for bipartite graphs.
    Example
    -------
    >>> from sknetwork.regression import Dirichlet
    >>> from sknetwork.data import house
    >>> dirichlet = Dirichlet()
    >>> adjacency = house()
    >>> seeds = {0: 1, 2: 0}
    >>> values = dirichlet.fit_predict(adjacency, seeds)
    >>> np.round(values, 2)
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
            seeds: Optional[Union[dict, np.ndarray]] = None, seeds_row: Optional[Union[dict, np.ndarray]] = None,
            seeds_col: Optional[Union[dict, np.ndarray]] = None, init: Optional[float] = None,
            force_bipartite: bool = False) -> 'Dirichlet':
        """Compute the solution to the Dirichlet problem (temperatures at equilibrium).

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        seeds :
            Temperatures of seed nodes (dictionary or vector). Negative temperatures ignored.
        seeds_row, seeds_col :
            Temperatures of rows and columns for bipartite graphs. Negative temperatures ignored.
        init :
            Temperature of non-seed nodes in initial state.
            If ``None``, use the average temperature of seed nodes (default).
        force_bipartite :
            If ``True``, consider the input matrix as a biadjacency matrix (default = ``False``).

        Returns
        -------
        self: :class:`Dirichlet`
        """
        adjacency, seeds, self.bipartite = get_adjacency_seeds(input_matrix, force_bipartite=force_bipartite,
                                                               seeds=seeds, seeds_row=seeds_row, seeds_col=seeds_col)
        temperatures, border = init_temperatures(seeds, init)
        values = temperatures.copy()
        diffusion = normalize(adjacency)
        for i in range(self.n_iter):
            values = diffusion.dot(values)
            values[border] = temperatures[border]

        self.values_ = values
        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self
