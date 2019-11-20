#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 17 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

from typing import Union, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, spsolve

from sknetwork.basics.rand_walk import transition_matrix
from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.checks import check_format, is_square


def limit_conditions(personalization: Union[np.ndarray, dict], n: int) -> Tuple:
    """Compute personalization vector and border indicator.

    Parameters
    ----------
    personalization:
        Array or dictionary indicating the fixed temperatures in the graph.
    n:
        Number of nodes in the graph.

    Returns
    -------
    b:
        Personalization vector.
    border:
        Border boolean indicator.

    """

    b: np.ndarray = np.zeros(n)
    border: np.ndarray = np.zeros(n)
    if type(personalization) == dict:
        b[list(personalization.keys())] = list(personalization.values())
        border[list(personalization.keys())] = 1
    elif type(personalization) == np.ndarray and len(personalization) == n:
        b = personalization
        border = personalization.astype(bool)
    else:
        raise ValueError('Personalization must be a dictionary or a vector'
                         ' of length equal to the number of nodes.')

    return b, border


class Diffusion(BaseRanking):
    """
    Computes the temperature of each node, associated with the diffusion along the edges (heat equation).

    Parameters
    ----------
    solver : str
        Which solver to use: 'spsolve' or 'lsqr' (default).

    Attributes
    ----------
    score_ : np.ndarray
        Score of each node (= temperature). Only raw nodes in the case of bipartite inputs.

    Example
    -------
    >>> from sknetwork.toy_graphs import house
    >>> diffusion = Diffusion(solver='spsolve')
    >>> adjacency = house()
    >>> personalization = {0: 0, 1: 1}
    >>> np.round(diffusion.fit(adjacency, personalization).score_, 2)
    array([0.  , 1.  , 0.86, 0.71, 0.57])

    References
    ----------
    Chung, F. (2007). The heat kernel as the pagerank of a graph. Proceedings of the National Academy of Sciences.
    """

    def __init__(self, solver: str = 'lsqr'):
        super(Diffusion, self).__init__()

        self.solver = solver

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
            raise ValueError('The adjacency matrix should be square. Consider using '
                             'sknetwork.utils.adjacency_format.bipartite2undirected.')

        b, border = limit_conditions(personalization, n)
        interior: sparse.csr_matrix = sparse.diags(1 - border, shape=(n, n), format='csr')
        diffusion_matrix = interior.dot(transition_matrix(adjacency))

        a = sparse.eye(n, format='csr') - diffusion_matrix
        if self.solver == 'spsolve':
            score = spsolve(a, b)
        elif self.solver == 'lsqr':
            score = lsqr(a, b)[0]
        else:
            raise ValueError('Unknown solver.')
        self.score_ = np.clip(score, np.min(b), np.max(b))

        return self


class BiDiffusion(BaseRanking):
    """Diffusion ranking algorithm on the normalized cocitation graph.

    See :class:`sknetwork.basics.co_neighbors_graph`

    """

    def __init__(self):
        super(BiDiffusion, self).__init__()

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
        bckward: sparse.csr_matrix = interior.dot(transition_matrix(biadjacency))
        forward: sparse.csr_matrix = transition_matrix(biadjacency.T)

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
            return x - bckward.dot(forward.dot(x))

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
            return x - forward.T.dot(bckward.T.dot(x))

        # noinspection PyArgumentList
        a = sparse.linalg.LinearOperator(dtype=float, shape=(n1, n1), matvec=mv, rmatvec=rmv)
        # noinspection PyTypeChecker
        score = lsqr(a, b)[0]
        self.score_ = np.clip(score, np.min(b), np.max(b))

        return self
