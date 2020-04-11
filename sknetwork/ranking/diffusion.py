#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 17 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

from typing import Union, Tuple, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab
from sknetwork.linalg.normalization import normalize
from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.check import check_format, check_seeds, is_square
from sknetwork.utils.format import bipartite2undirected
from sknetwork.utils.seeds import stack_seeds
from sknetwork.utils.verbose import VerboseMixin


def limit_conditions(personalization: np.ndarray) -> Tuple:
    """Compute seeds vector and border indicator.

    Parameters
    ----------
    personalization:
        Array or dictionary indicating the fixed temperatures in the graph.
        In order to avoid ambiguities, temperatures must be non-negative.

    Returns
    -------
    b:
        Personalization vector.
    border:
        Border boolean indicator.

    """
    b = personalization
    border = (b >= 0)
    b[~border] = 0
    return b.astype(float), border.astype(bool)


class Diffusion(BaseRanking, VerboseMixin):
    """Temperature of each node, associated with the diffusion along the edges (heat equation).

    * Graphs
    * Digraphs

    Parameters
    ----------
    n_iter: int
        If ``n_iter > 0``, simulate the diffusion in discrete time for n_iter steps.
        If ``n_iter <= 0``, use BIConjugate Gradient STABilized iteration
        to solve the Dirichlet problem.
    verbose: bool
        Verbose mode.

    Attributes
    ----------
    scores_ : np.ndarray
        Score of each node (= temperature).

    Example
    -------
    >>> from sknetwork.ranking import Diffusion
    >>> from sknetwork.data import house
    >>> diffusion = Diffusion()
    >>> adjacency = house()
    >>> seeds = {0: 1, 2: 0}
    >>> scores = diffusion.fit_transform(adjacency, seeds)
    >>> np.round(scores, 2)
    array([1.  , 0.54, 0.  , 0.31, 0.62])

    References
    ----------
    Chung, F. (2007). The heat kernel as the pagerank of a graph. Proceedings of the National Academy of Sciences.
    """

    def __init__(self, n_iter: int = 0, verbose: bool = False):
        super(Diffusion, self).__init__()
        VerboseMixin.__init__(self, verbose)

        self.n_iter = n_iter

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray],
            seeds: Optional[Union[dict, np.ndarray]] = None, initial_state: Optional = None) -> 'Diffusion':
        """Compute the diffusion (temperature at equilibrium).

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.
        seeds :
            Temperatures of border nodes (dictionary or vector). Negative temperatures ignored.
        initial_state :
            Initial state of temperatures.

        Returns
        -------
        self: :class:`Diffusion`
        """
        adjacency = check_format(adjacency)
        n: int = adjacency.shape[0]
        if not is_square(adjacency):
            raise ValueError('The adjacency matrix should be square. See BiDiffusion.')
        if seeds is None:
            self.scores_ = np.ones(n) / n
            return self

        seeds = check_seeds(seeds, n)
        b, border = limit_conditions(seeds)
        tmin, tmax = np.min(b[border]), np.max(b)

        interior: sparse.csr_matrix = sparse.diags(~border, shape=(n, n), format='csr', dtype=float)
        diffusion_matrix = interior.dot(normalize(adjacency))

        if initial_state is None:
            if tmin != tmax:
                initial_state = b[border].mean() * np.ones(n)
            else:
                initial_state = np.zeros(n)
            initial_state[border] = b[border]

        if self.n_iter > 0:
            scores = initial_state
            for i in range(self.n_iter):
                scores = diffusion_matrix.dot(scores)
                scores[border] = b[border]

        else:
            a = sparse.eye(n, format='csr', dtype=float) - diffusion_matrix
            scores, info = bicgstab(a, b, atol=0., x0=initial_state)
            self._scipy_solver_info(info)

        if tmin != tmax:
            self.scores_ = np.clip(scores, tmin, tmax)
        else:
            self.scores_ = scores
        return self


class BiDiffusion(Diffusion):
    """Temperature of each node of a bipartite graph, associated with the diffusion along the edges (heat equation).

    * Bigraphs

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
    >>> bidiffusion = BiDiffusion()
    >>> biadjacency = star_wars()
    >>> seeds = {0: 1, 2: 0}
    >>> scores = bidiffusion.fit_transform(biadjacency, seeds)
    >>> np.round(scores, 2)
    array([1.  , 0.5 , 0.  , 0.29])
    """

    def __init__(self, n_iter: int = 0, verbose: bool = False):
        super(BiDiffusion, self).__init__(n_iter, verbose)

        self.scores_row_ = None
        self.scores_col_ = None

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
            Initial state of temperatures.

        Returns
        -------
        self: :class:`BiDiffusion`
        """
        biadjacency = check_format(biadjacency)
        n_row, n_col = biadjacency.shape
        seeds = stack_seeds(n_row, n_col, seeds_row, seeds_col)

        adjacency = bipartite2undirected(biadjacency)
        Diffusion.fit(self, adjacency, seeds)

        self.scores_row_ = self.scores_[:n_row]
        self.scores_col_ = self.scores_[n_row:]
        self.scores_ = self.scores_row_

        return self
