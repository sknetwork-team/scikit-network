#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Union, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab, LinearOperator

from sknetwork.linalg.normalization import normalize
from sknetwork.regression.base import BaseRegressor
from sknetwork.utils.check import check_is_proba
from sknetwork.utils.format import get_adjacency_seeds
from sknetwork.utils.verbose import VerboseMixin


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


class DirichletOperator(LinearOperator):
    """Diffusion in discrete time as a LinearOperator.

    Parameters
    ----------
    adjacency : sparse.csr_matrix
        Adjacency matrix of the graph.
    damping_factor : float
        Damping factor.
    border : np.ndarray (bool)
        Border nodes. If ``None``, then the diffusion is free.

    Attributes
    ----------
    a : sparse.csr_matrix
        Diffusion matrix.
    b : np.ndarray
        Regularization (uniform).
    """
    def __init__(self, adjacency: sparse.csr_matrix, damping_factor: float, border: np.ndarray = None):
        super(DirichletOperator, self).__init__(shape=adjacency.shape, dtype=float)
        n = adjacency.shape[0]
        out_nodes = adjacency.dot(np.ones(n)).astype(bool)
        if border is None:
            border = np.zeros(n, dtype=bool)
        interior: sparse.csr_matrix = sparse.diags(~border, shape=(n, n), format='csr', dtype=float)
        self.a = damping_factor * interior.dot(normalize(adjacency))
        self.b = interior.dot(np.ones(n) - damping_factor * out_nodes) / n

    def _matvec(self, x: np.ndarray):
        return self.a.dot(x) + self.b * x.sum()


class DeltaDirichletOperator(DirichletOperator):
    """Diffusion in discrete time as a LinearOperator (delta of temperature).

    Parameters
    ----------
    adjacency : sparse.csr_matrix
        Adjacency matrix of the graph.
    damping_factor : float
        Damping factor.
    border : np.ndarray (bool)
        Border nodes. If ``None``, then the diffusion is free.

    Attributes
    ----------
    a : sparse.csr_matrix
        Diffusion matrix.
    b : np.ndarray
        Regularization (uniform).
    """
    def __init__(self, adjacency: sparse.csr_matrix, damping_factor: float, border: np.ndarray = None):
        super(DeltaDirichletOperator, self).__init__(adjacency, damping_factor, border)

    def _matvec(self, x: np.ndarray):
        return self.a.dot(x) + self.b * x.sum() - x


class Diffusion(BaseRegressor):
    """Regression by diffusion along the edges (heat equation).

    Parameters
    ----------
    n_iter : int
        Number of steps of the diffusion in discrete time (must be positive).
    damping_factor : float (optional)
        Damping factor (default value = 1).

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
    >>> values = diffusion.fit_transform(adjacency, seeds)
    >>> np.round(values, 2)
    array([0.58, 0.56, 0.38, 0.58, 0.42])

    References
    ----------
    Chung, F. (2007). The heat kernel as the pagerank of a graph. Proceedings of the National Academy of Sciences.
    """
    def __init__(self, n_iter: int = 3, damping_factor: Optional[float] = None):
        super(Diffusion, self).__init__()

        if n_iter <= 0:
            raise ValueError('The number of iterations must be positive.')
        else:
            self.n_iter = n_iter
        if damping_factor is None:
            damping_factor = 1.
        check_is_proba(damping_factor, 'Damping factor')
        self.damping_factor = damping_factor
        self.bipartite = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray],
            seeds: Optional[Union[dict, np.ndarray]] = None, seeds_row: Optional[Union[dict, np.ndarray]] = None,
            seeds_col: Optional[Union[dict, np.ndarray]] = None, init: Optional[float] = None) \
            -> 'Diffusion':
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

        Returns
        -------
        self: :class:`Diffusion`
        """
        adjacency, seeds, self.bipartite = get_adjacency_seeds(input_matrix, allow_directed=True, seeds=seeds,
                                                               seeds_row=seeds_row, seeds_col=seeds_col)
        values, _ = init_temperatures(seeds, init)
        diffusion = DirichletOperator(adjacency, self.damping_factor)
        for i in range(self.n_iter):
            values = diffusion.dot(values)

        self.values_ = values
        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self


class Dirichlet(BaseRegressor, VerboseMixin):
    """Regression by the Dirichlet problem (heat diffusion with boundary constraints).

    Parameters
    ----------
    n_iter : int
        If positive, number of steps of the diffusion in discrete time.
        Otherwise, solve the Dirichlet problem by the bi-conjugate gradient stabilized method.
    damping_factor : float (optional)
        Damping factor (default value = 1).
    verbose : bool
        Verbose mode.

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
    >>> values = dirichlet.fit_transform(adjacency, seeds)
    >>> np.round(values, 2)
    array([1.  , 0.54, 0.  , 0.31, 0.62])

    References
    ----------
    Chung, F. (2007). The heat kernel as the pagerank of a graph. Proceedings of the National Academy of Sciences.
    """
    def __init__(self, n_iter: int = 10, damping_factor: Optional[float] = None, verbose: bool = False):
        super(Dirichlet, self).__init__()
        VerboseMixin.__init__(self, verbose)

        self.n_iter = n_iter
        if damping_factor is None:
            damping_factor = 1.
        check_is_proba(damping_factor, 'Damping factor')
        self.damping_factor = damping_factor
        self.bipartite = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray],
            seeds: Optional[Union[dict, np.ndarray]] = None, seeds_row: Optional[Union[dict, np.ndarray]] = None,
            seeds_col: Optional[Union[dict, np.ndarray]] = None, init: Optional[float] = None) -> 'Dirichlet':
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

        Returns
        -------
        self: :class:`Dirichlet`
        """
        adjacency, seeds, self.bipartite = get_adjacency_seeds(input_matrix, seeds=seeds, seeds_row=seeds_row,
                                                               seeds_col=seeds_col)
        values, border = init_temperatures(seeds, init)
        if self.n_iter > 0:
            diffusion = DirichletOperator(adjacency, self.damping_factor, border)
            for i in range(self.n_iter):
                values = diffusion.dot(values)
                values[border] = seeds[border]
        else:
            a = DeltaDirichletOperator(adjacency, self.damping_factor, border)
            b = -seeds
            b[~border] = 0
            values, info = bicgstab(a, b, atol=0., x0=values)
            self._scipy_solver_info(info)

        temp_min, temp_max = seeds[border].min(), seeds[border].max()
        self.values_ = np.clip(values, temp_min, temp_max)
        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self
