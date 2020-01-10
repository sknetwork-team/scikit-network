#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 17 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

import warnings
from typing import Union, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab
from sknetwork.basics.rand_walk import transition_matrix
from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.checks import check_format, is_square
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
            warnings.warn(Warning("Negative temperatures will be ignored"))

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
    verbose: bool
        Verbose mode.
    n_iter: int
        If ``n_iter > 0``, the algorithm will emulate the diffusion for n_iter steps.
        If ``n_iter <= 0``, the algorithm will use BIConjugate Gradient STABilized iteration
        to solve the Dirichlet problem.

    Attributes
    ----------
    scores_ : np.ndarray
        Score of each node (= temperature).

    Example
    -------
    >>> from sknetwork.data import house
    >>> diffusion = Diffusion()
    >>> adjacency = house()
    >>> personalization = {4: 0.25, 1: 1}
    >>> np.round(diffusion.fit_transform(adjacency, personalization), 2)
    array([0.62, 1.  , 0.75, 0.5 , 0.25])

    References
    ----------
    Chung, F. (2007). The heat kernel as the pagerank of a graph. Proceedings of the National Academy of Sciences.
    """

    def __init__(self, verbose: bool = False, n_iter: int = 0):
        super(Diffusion, self).__init__()
        VerboseMixin.__init__(self, verbose)

        self.n_iter = n_iter

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray],
            personalization: Union[dict, np.ndarray]) -> 'Diffusion':
        """
        Compute the diffusion (temperature at equilibrium).

        Parameters
        ----------
        adjacency :
            Adjacency or biadjacency matrix of the graph.
        personalization :
            Dictionary or vector (temperature of border nodes).

        Returns
        -------
        self: :class:`Diffusion`
        """
        adjacency = check_format(adjacency)
        n: int = adjacency.shape[0]
        if not is_square(adjacency):
            raise ValueError('The adjacency matrix should be square. See BiDiffusion.')

        b, border = limit_conditions(personalization, n)
        interior: sparse.csr_matrix = sparse.diags(~border, shape=(n, n), format='csr', dtype=float)
        diffusion_matrix = interior.dot(transition_matrix(adjacency))

        x0 = b[border].mean() * np.ones(n)
        x0[border] = b[border]

        if self.n_iter > 0:
            scores = x0
            for i in range(self.n_iter):
                scores = diffusion_matrix.dot(scores)
                scores[border] = b[border]

        else:
            a = sparse.eye(n, format='csr', dtype=float) - diffusion_matrix
            scores, info = bicgstab(a, b, atol=0., x0=x0)
            self.scipy_solver_info(info)

        self.scores_ = np.clip(scores, np.min(b[border]), np.max(b))
        return self


class BiDiffusion(Diffusion):
    """Compute the temperature of each node of a bipartite graph,
    associated with the diffusion along the edges (heat equation).

    Attributes
    ----------
    row_scores_ : np.ndarray
        Scores of rows.
    col_scores_ : np.ndarray
        Scores of columns.
    scores_ : np.ndarray
        Scores of all nodes (concatenation of scores of rows and scores of columns).

    Example
    -------
    >>> from sknetwork.data import star_wars_villains
    >>> bidiffusion = BiDiffusion()
    >>> biadjacency: sparse.csr_matrix = star_wars_villains()
    >>> biadjacency.shape
    (4, 3)
    >>> len(bidiffusion.fit_transform(biadjacency, {0: 1, 1: 0}))
    7
    """

    def __init__(self, verbose: bool = False, n_iter: int = 0):
        super(BiDiffusion, self).__init__(verbose, n_iter)

        self.row_scores_ = None
        self.col_scores_ = None

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray],
            personalization: Union[dict, np.ndarray]) -> 'BiDiffusion':
        """
        Compute the diffusion (temperature at equilibrium).

        Parameters
        ----------
        biadjacency :
            Adjacency or biadjacency matrix of the graph.
        personalization :
            Dictionary or vector (temperature of border nodes).

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

        x0 = np.zeros(n1)
        ix = (b >= 0)
        x0[ix] = b[ix]

        if self.n_iter > 0:
            scores = x0
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
            scores, info = bicgstab(a, x0, atol=0., x0=x0)
            self.scipy_solver_info(info)

        self.row_scores_ = np.clip(scores, np.min(b), np.max(b))
        self.col_scores_ = transition_matrix(biadjacency.T).dot(self.row_scores_)
        self.scores_ = np.concatenate((self.row_scores_, self.col_scores_))

        return self
