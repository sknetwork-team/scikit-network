#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 17 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Union, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab, LinearOperator

from sknetwork.linalg.normalization import normalize
from sknetwork.ranking.base import BaseRanking, BaseBiRanking
from sknetwork.utils.check import check_format, check_seeds, check_square, check_is_proba
from sknetwork.utils.format import bipartite2undirected
from sknetwork.utils.seeds import stack_seeds
from sknetwork.utils.verbose import VerboseMixin


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


class Diffusion(BaseRanking):
    """Temperature of each node, associated with the diffusion along the edges (heat equation).

    * Graphs
    * Digraphs

    Parameters
    ----------
    n_iter : int
        Number of steps of the diffusion in discrete time (must be positive).
    damping_factor : float (optional)
        Damping factor (default value = 1).

    Attributes
    ----------
    scores_ : np.ndarray
        Score of each node (= temperature).

    Example
    -------
    >>> from sknetwork.data import house
    >>> diffusion = Diffusion(n_iter=2)
    >>> adjacency = house()
    >>> seeds = {0: 1, 2: 0}
    >>> scores = diffusion.fit_transform(adjacency, seeds)
    >>> np.round(scores, 2)
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

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Optional[Union[dict, np.ndarray]] = None,
            initial_state: Optional = None) -> 'Diffusion':
        """Compute the diffusion (temperature at equilibrium).

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.
        seeds :
            Temperatures of border nodes (dictionary or vector). Negative temperatures ignored.
        initial_state :
            Temperatures in initial state (optional).

        Returns
        -------
        self: :class:`Diffusion`
        """
        adjacency = check_format(adjacency)
        check_square(adjacency)
        n: int = adjacency.shape[0]
        if seeds is None:
            self.scores_ = np.ones(n) / n
            return self

        seeds = check_seeds(seeds, n)
        border = (seeds >= 0)

        if initial_state is None:
            initial_state = seeds[border].mean() * np.ones(n)
        initial_state[border] = seeds[border]

        scores = initial_state
        diffusion = DirichletOperator(adjacency, self.damping_factor)
        for i in range(self.n_iter):
            scores = diffusion.dot(scores)

        tmin, tmax = seeds[border].min(), seeds[border].max()
        self.scores_ = np.clip(scores, tmin, tmax)

        return self


class BiDiffusion(Diffusion, BaseBiRanking):
    """Temperature of each node of a bipartite graph, associated with the diffusion along the edges (heat equation).

    * Bigraphs

    Parameters
    ----------
    n_iter : int
        Number of steps of the diffusion in discrete time (must be positive).
    damping_factor : float (optional)
        Damping factor (default value = 1).

    Attributes
    ----------
    scores_ : np.ndarray
        Scores of rows.
    scores_row_ : np.ndarray
        Scores of rows (copy of **scores_**).
    scores_col_ : np.ndarray
        Scores of columns.

    Example
    -------
    >>> from sknetwork.ranking import BiDiffusion
    >>> from sknetwork.data import star_wars
    >>> bidiffusion = BiDiffusion(n_iter=2)
    >>> biadjacency = star_wars()
    >>> scores = bidiffusion.fit_transform(biadjacency, seeds_row = {0: 1, 2: 0})
    >>> np.round(scores, 2)
    array([0.5 , 0.5 , 0.42, 0.38])
    """
    def __init__(self, n_iter: int = 3, damping_factor: Optional[float] = None):
        super(BiDiffusion, self).__init__(n_iter, damping_factor)

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray],
            seeds_row: Optional[Union[dict, np.ndarray]] = None, seeds_col: Optional[Union[dict, np.ndarray]] = None,
            initial_state: Optional = None) -> 'BiDiffusion':
        """Compute the diffusion (temperature at equilibrium).

        Parameters
        ----------
        biadjacency :
            Biadjacency matrix, shape (n_row, n_col).
        seeds_row :
            Temperatures of row border nodes (dictionary or vector of size n_row). Negative temperatures ignored.
        seeds_col :
            Temperatures of column border nodes (dictionary or vector of size n_row). Negative temperatures ignored.
        initial_state :
            Temperatures in initial state.

        Returns
        -------
        self: :class:`BiDiffusion`
        """
        biadjacency = check_format(biadjacency)
        n_row, n_col = biadjacency.shape
        seeds = stack_seeds(n_row, n_col, seeds_row, seeds_col)

        adjacency = bipartite2undirected(biadjacency)
        Diffusion.fit(self, adjacency, seeds)
        self._split_vars(n_row)

        return self


class Dirichlet(BaseRanking, VerboseMixin):
    """Ranking by the Dirichlet problem (heat diffusion with boundary constraints).

    * Graphs
    * Digraphs

    Parameters
    ----------
    n_iter : int
        If positive, number of steps of the diffusion in discrete time.
        Otherwise, solve the Dirichlet problem by the BIConjugate Gradient STABilized method.
    damping_factor : float (optional)
        Damping factor (default value = 1).
    verbose : bool
        Verbose mode.

    Attributes
    ----------
    scores_ : np.ndarray
        Score of each node (= temperature).

    Example
    -------
    >>> from sknetwork.ranking import Dirichlet
    >>> from sknetwork.data import house
    >>> dirichlet = Dirichlet()
    >>> adjacency = house()
    >>> seeds = {0: 1, 2: 0}
    >>> scores = dirichlet.fit_transform(adjacency, seeds)
    >>> np.round(scores, 2)
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

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Optional[Union[dict, np.ndarray]] = None,
            initial_state: Optional = None) -> 'Dirichlet':
        """Compute the solution to the Dirichlet problem (temperature at equilibrium).

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.
        seeds :
            Temperatures of border nodes (dictionary or vector). Negative temperatures ignored.
        initial_state :
            Temperatures in initial state (optional).

        Returns
        -------
        self: :class:`Dirichlet`
        """
        adjacency = check_format(adjacency)
        check_square(adjacency)
        n: int = adjacency.shape[0]
        if seeds is None:
            self.scores_ = np.ones(n) / n
            return self

        seeds = check_seeds(seeds, n)
        border = (seeds >= 0)

        if initial_state is None:
            initial_state = seeds[border].mean() * np.ones(n)
        initial_state[border] = seeds[border]

        if self.n_iter > 0:
            scores = initial_state
            diffusion = DirichletOperator(adjacency, self.damping_factor, border)
            for i in range(self.n_iter):
                scores = diffusion.dot(scores)
                scores[border] = seeds[border]
        else:
            a = DeltaDirichletOperator(adjacency, self.damping_factor, border)
            b = -seeds
            b[~border] = 0
            scores, info = bicgstab(a, b, atol=0., x0=initial_state)
            self._scipy_solver_info(info)

        tmin, tmax = seeds[border].min(), seeds[border].max()
        self.scores_ = np.clip(scores, tmin, tmax)

        return self


class BiDirichlet(Dirichlet, BaseBiRanking):
    """Ranking by the Dirichlet problem in bipartite graphs (heat diffusion with boundary constraints).

    * Bigraphs

    Parameters
    ----------
    n_iter : int
        If positive, number of steps of the diffusion in discrete time.
        Otherwise, solve the Dirichlet problem by the BIConjugate Gradient STABilized method.
    damping_factor : float (optional)
        Damping factor (default value = 1).
    verbose : bool
        Verbose mode.

    Attributes
    ----------
    scores_ : np.ndarray
        Scores of rows.
    scores_row_ : np.ndarray
        Scores of rows (copy of **scores_**).
    scores_col_ : np.ndarray
        Scores of columns.

    Example
    -------
    >>> from sknetwork.ranking import BiDirichlet
    >>> from sknetwork.data import star_wars
    >>> bidirichlet = BiDirichlet()
    >>> biadjacency = star_wars()
    >>> scores = bidirichlet.fit_transform(biadjacency, seeds_row = {0: 1, 2: 0})
    >>> np.round(scores, 2)
    array([1.  , 0.5 , 0.  , 0.29])
    """

    def __init__(self, n_iter: int = 10, damping_factor: Optional[float] = None, verbose: bool = False):
        super(BiDirichlet, self).__init__(n_iter, damping_factor, verbose)

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray],
            seeds_row: Optional[Union[dict, np.ndarray]] = None, seeds_col: Optional[Union[dict, np.ndarray]] = None,
            initial_state: Optional = None) -> 'BiDirichlet':
        """Compute the solution to the Dirichlet problem (temperature at equilibrium).

        Parameters
        ----------
        biadjacency :
            Biadjacency matrix, shape (n_row, n_col).
        seeds_row :
            Temperatures of row border nodes (dictionary or vector of size n_row). Negative temperatures ignored.
        seeds_col :
            Temperatures of column border nodes (dictionary or vector of size n_row). Negative temperatures ignored.
        initial_state :
            Temperatures in initial state.

        Returns
        -------
        self: :class:`BiDirichlet`
        """
        biadjacency = check_format(biadjacency)
        n_row, n_col = biadjacency.shape
        seeds = stack_seeds(n_row, n_col, seeds_row, seeds_col)

        adjacency = bipartite2undirected(biadjacency)
        Dirichlet.fit(self, adjacency, seeds)
        self._split_vars(n_row)

        return self
