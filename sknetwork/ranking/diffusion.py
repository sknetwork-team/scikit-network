#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 17 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, spsolve

from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format, is_square


class Diffusion(Algorithm):
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
    col_score_ : np.ndarray
        Score of each column node (= temperature) for bipartite inputs.

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
        self.solver = solver

        self.score_ = None
        self.col_score_ = None

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
        n1: int = adjacency.shape[0]
        if not is_square(adjacency):
            raise ValueError('The adjacency matrix should be square. Consider using '
                             'sknetwork.utils.adjacency_format.bipartite2undirected.')
        n: int = adjacency.shape[0]

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

        diag_out: sparse.csr_matrix = sparse.diags(adjacency.dot(np.ones(n)), shape=(n, n), format='csr')
        diag_out.data = 1 / diag_out.data
        interior: sparse.csr_matrix = sparse.diags(1 - border, shape=(n, n), format='csr')
        diffusion_matrix = interior.dot(diag_out.dot(adjacency))

        a = sparse.eye(n, format='csr') - diffusion_matrix
        if self.solver == 'spsolve':
            score = spsolve(a, b)
        elif self.solver == 'lsqr':
            score = lsqr(a, b)[0]
        else:
            raise ValueError('Unknown solver.')
        score = np.clip(score, np.min(b), np.max(b))

        if n1 == n:
            self.score_ = score
        else:
            self.score_ = score[:n1]
            self.col_score_ = score[n1:]

        return self
