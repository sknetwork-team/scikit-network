#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 31 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Union
from sknetwork.utils.checks import check_format, has_nonnegative_entries, is_square
from sknetwork.utils.algorithm_base_class import Algorithm


class PageRank(Algorithm):
    """
    Computes the PageRank of each node, corresponding to its frequency of visit by a random walk.

    The random walk restarts with some fixed probability. The restart distribution can be personalized by the user.
    This variant is known as Personalized PageRank.

    Parameters
    ----------
    damping_factor : float
        Probability to continue the random walk.

    Attributes
    ----------
    score_ : np.ndarray
        PageRank score of each node.

    Example
    -------
    >>> from sknetwork.toy_graphs import rock_paper_scissors
    >>> pagerank = PageRank()
    >>> adjacency = rock_paper_scissors()
    >>> np.round(pagerank.fit(adjacency).score_, 2)
    array([0.33, 0.33, 0.33])

    References
    ----------
    Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web.
    Stanford InfoLab.
    """
    def __init__(self, damping_factor: float = 0.85):
        self.score_ = None
        if damping_factor < 0 or damping_factor >= 1:
            raise ValueError('Damping factor must be between 0 and 1.')
        else:
            self.damping_factor = damping_factor

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray],
            personalization: Union[None, np.ndarray] = None) -> 'PageRank':
        """Standard PageRank via matrix factorization or D-iteration.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph.
        personalization :
            If ``None``, the uniform distribution is used.
            Otherwise, a non-negative, non-zero vector must be provided.

        Returns
        -------
        self: :class: 'PageRank'
        """
        adjacency = check_format(adjacency)
        if not is_square(adjacency):
            raise ValueError('The adjacency matrix must be square.')
        else:
            n: int = adjacency.shape[0]

        diag_out: sparse.csr_matrix = sparse.diags(adjacency.dot(np.ones(n)), shape=(n, n), format='csr')
        diag_out.data = 1 / diag_out.data
        transition_matrix = diag_out.dot(adjacency)

        if personalization is None:
            restart_prob: np.ndarray = np.ones(n) / n
        else:
            if has_nonnegative_entries(personalization) and len(personalization) == n and np.sum(personalization):
                restart_prob = personalization.astype(float) / np.sum(personalization)
            else:
                raise ValueError('Personalization must be None or a non-negative, non-null vector.')

        a = sparse.eye(n, format='csr') - self.damping_factor * transition_matrix.T
        b = (1 - self.damping_factor * diag_out.data.astype(bool)) * restart_prob
        x = spsolve(a, b)

        self.score_ = abs(x.real) / abs(x.real).sum()

        return self
