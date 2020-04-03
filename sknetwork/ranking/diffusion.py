#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 17 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

import warnings
from typing import Union, Tuple, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab
from sknetwork.basics.rand_walk import transition_matrix
from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.check import check_format, is_square
from sknetwork.utils.verbose import VerboseMixin


def limit_conditions(personalization: Union[np.ndarray, dict], n: int) -> Tuple:
    """Compute personalization vector and border indicator.

    Parameters
    ----------
    personalization:
        Array or dictionary indicating the fixed temperatures in the graph.
        In order to avoid ambiguities, temperatures must be non-negative.
    n:
        Number of nodes in the graph.

    Returns
    -------
    b:
        Personalization vector.
    border:
        Border boolean indicator.

    """

    if type(personalization) == dict:
        keys = np.array(list(personalization.keys()))
        vals = np.array(list(personalization.values()))
        if np.min(vals) < 0:
            warnings.warn(Warning("Negative temperatures will be ignored."))

        ix = (vals >= 0)
        keys = keys[ix]
        vals = vals[ix]

        b = -np.ones(n)
        b[keys] = vals

    elif type(personalization) == np.ndarray and len(personalization) == n:
        b = personalization
    else:
        raise ValueError('Personalization must be a dictionary or a vector'
                         ' of length equal to the number of nodes.')

    border = (b >= 0)
    b[~border] = 0
    return b.astype(float), border.astype(bool)


class Diffusion(BaseRanking, VerboseMixin):
    """
    Computes the temperature of each node, associated with the diffusion along the edges (heat equation).

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
    >>> from sknetwork.data import house
    >>> diffusion = Diffusion()
    >>> adjacency = house()
    >>> personalization = {0: 1, 2: 0}
    >>> np.round(diffusion.fit_transform(adjacency, personalization), 2)
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
            personalization: Union[dict, np.ndarray], initial_state: Optional = None) -> 'Diffusion':
        """
        Compute the diffusion (temperature at equilibrium).

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.
        personalization :
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

        b, border = limit_conditions(personalization, n)
        tmin, tmax = np.min(b[border]), np.max(b)

        interior: sparse.csr_matrix = sparse.diags(~border, shape=(n, n), format='csr', dtype=float)
        diffusion_matrix = interior.dot(transition_matrix(adjacency))

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
    """Compute the temperature of each node of a bipartite graph,
    associated with the diffusion along the edges (heat equation).

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
    >>> from sknetwork.data import star_wars
    >>> bidiffusion = BiDiffusion()
    >>> biadjacency = star_wars()
    >>> personalization = {0: 1, 2: 0}
    >>> np.round(bidiffusion.fit_transform(biadjacency, personalization), 2)
    array([1.  , 0.5 , 0.  , 0.29])
    """

    def __init__(self, n_iter: int = 0, verbose: bool = False):
        super(BiDiffusion, self).__init__(n_iter, verbose)

        self.scores_row_ = None
        self.scores_col_ = None

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray],
            personalization: Union[dict, np.ndarray], initial_state: Optional = None) -> 'BiDiffusion':
        """
        Compute the diffusion (temperature at equilibrium).

        Parameters
        ----------
        biadjacency :
            Biadjacency matrix, shape (n1, n2).
        personalization :
            Temperatures of border nodes (dictionary or vector of size n1). Negative temperatures ignored.
        initial_state :
            Initial state of temperatures.

        Returns
        -------
        self: :class:`BiDiffusion`
        """
        biadjacency = check_format(biadjacency)
        n1, n2 = biadjacency.shape

        b, border = limit_conditions(personalization, n1)
        interior: sparse.csr_matrix = sparse.diags(1 - border, format='csr')
        backward: sparse.csr_matrix = interior.dot(transition_matrix(biadjacency))
        forward: sparse.csr_matrix = transition_matrix(biadjacency.T)

        initial_state = np.zeros(n1)
        ix = (b >= 0)
        initial_state[ix] = b[ix]

        if self.n_iter > 0:
            scores = initial_state
            for i in range(self.n_iter):
                scores = backward.dot(forward.dot(scores))
                scores[border] = b[border]

        else:

            def mv(x: np.ndarray) -> np.ndarray:
                """Matrix vector multiplication for BiDiffusion operator.

                Parameters
                ----------
                x:
                    vector

                Returns
                -------
                matrix-vector product

                """
                return x - backward.dot(forward.dot(x))

            def rmv(x: np.ndarray) -> np.ndarray:
                """Matrix vector multiplication for transposed BiDiffusion operator.

                Parameters
                ----------
                x:
                    vector

                Returns
                -------
                matrix-vector product

                """
                return x - forward.T.dot(backward.T.dot(x))

            # noinspection PyArgumentList
            a = sparse.linalg.LinearOperator(dtype=float, shape=(n1, n1), matvec=mv, rmatvec=rmv)
            # noinspection PyTypeChecker
            scores, info = bicgstab(a, initial_state, atol=0., x0=initial_state)
            self._scipy_solver_info(info)

        self.scores_row_ = np.clip(scores, np.min(b), np.max(b))
        self.scores_col_ = transition_matrix(biadjacency.T).dot(self.scores_row_)
        self.scores_ = self.scores_row_

        return self
