#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 31 2019
@author: Nathan de Lara <ndelara@enst.fr>
@author: Fabien Mathieu
"""

import numpy as np
from sknetwork import njit, prange
from sknetwork.utils.checks import check_format, is_proba_array, is_square
from sknetwork.utils.randomized_matrix_factorization import SparseLR, randomized_eig
from sknetwork.utils.algorithm_base_class import Algorithm
from scipy import sparse
from typing import Union


@njit(parallel=True)
def diffusion(indptr, indices, data, flow_history, current_flow, damping_factor):
    """Diffusion for D-iteration.

    Parameters
    ----------
    indptr
    indices
    data
    flow_history
    current_flow
    damping_factor

    Returns
    -------

    """
    active = np.nonzero(current_flow)[0]
    for a in prange(len(active)):
        i = active[a]
        tmp = current_flow[i]
        flow_history[i] += tmp
        outnodes = indices[indptr[i]:indptr[i + 1]]
        if outnodes.size:
            current_flow[outnodes] += (damping_factor * tmp / outnodes.size) * data[indptr[i]:indptr[i + 1]]
        current_flow[i] -= tmp


class PageRank(Algorithm):
    """
    Assign to each node the probability for a random surfer to be on that node in a stationary
    regime with teleportation.

    The set of nodes to which the random surfer teleports can be chosen by the user. This variant is referred to as
    Personalized PageRank.

    Parameters
    ----------
    damping_factor: float
        Probability to jump according to the graph transition matrix in the random walk.
    method: str
        Specifies the used method. Must be 'diter' or 'spectral'
    n_iter: int
        Number of iterations if method is 'diter'

    Attributes
    ----------
    ranking_: np.ndarray
        Ranking of each node.

    Example
    -------
    >>> from sknetwork.toy_graphs import rock_paper_scissors_graph
    >>> graph = rock_paper_scissors_graph()
    >>> pagerank = PageRank().fit(graph)
    >>> np.round(pagerank.ranking_, 2)
    array([0.33, 0.33, 0.33])

    References
    ----------
    * Hong, D., Huynh, T. D., & Mathieu, F. (2015). D-iteration: diffusion approach for solving pagerank.
      arXiv preprint arXiv:1501.06350.

    * Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web.
      Stanford InfoLab.
    """
    def __init__(self, damping_factor: float = 0.85, method: str = 'diter', n_iter: int = 25):
        self.ranking_ = None
        if damping_factor < 0. or damping_factor > 1.:
            raise ValueError('Damping factor must be between 0 and 1.')
        else:
            self.damping_factor = damping_factor

        if method != 'diter' and method != 'spectral':
            raise ValueError("Method must be 'diter' or 'spectral'.")
        else:
            self.method = method

        if type(n_iter) != int:
            raise TypeError("Number of iterations must be an integer.")
        elif n_iter <= 0:
            raise ValueError("Number of iterations must be positive.'")
        else:
            self.n_iter = n_iter

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray],
            personalization: Union[None, np.ndarray] = None) -> 'PageRank':
        """Standard pagerank via matrix factorization or D-iteration.

        Parameters
        ----------
        adjacency
            Adjacency matrix of the graph.
        personalization
            If ``None``, the uniform probability distribution over the nodes is used.
            Otherwise, the user must provide a valid probability vector.

        """
        adjacency = check_format(adjacency)
        if not is_square(adjacency):
            raise ValueError('Adjacency must be square.')
        else:
            n_nodes: int = adjacency.shape[0]

        # pseudo inverse square-root out-degree matrix
        diag_out: sparse.csr_matrix = sparse.diags(adjacency.dot(np.ones(n_nodes)),
                                                   shape=(n_nodes, n_nodes), format='csr')
        diag_out.data = 1 / diag_out.data

        transition_matrix = diag_out.dot(adjacency)

        if personalization is None:
            root: np.ndarray = np.ones(n_nodes) / n_nodes
        else:
            if is_proba_array(personalization) and len(personalization) == n_nodes:
                root = personalization.astype(float)
            else:
                raise ValueError('Personalization must be None or a valid probability array.')

        if self.method == 'diter':

            current_flow = root
            flow_history = -current_flow / 2

            for i in range(self.n_iter):
                diffusion(transition_matrix.indptr, transition_matrix.indices, transition_matrix.data,
                          flow_history, current_flow, self.damping_factor)

            self.ranking_ = abs(flow_history) / np.sum(abs(flow_history))

        elif self.method == 'spectral':

            weight_matrix = sparse.eye(n_nodes, format='csr') - self.damping_factor * diag_out.astype(bool)
            root = weight_matrix.dot(root)

            pagerank_matrix = SparseLR(self.damping_factor * transition_matrix, [(root, np.ones(n_nodes))])
            _, eigenvector = randomized_eig(pagerank_matrix, 1)

            self.ranking_ = abs(eigenvector.real) / abs(eigenvector.real).sum()
            self.ranking_.reshape((self.ranking_.shape[0]))

        return self
