#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Sep 13 2018

Authors:
Thomas Bonald <thomas.bonald@telecom-paristech.fr>
Nathan De Lara <nathan.delara@telecom-paristech.fr>
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sknetwork.utils import safe_sparse_dot, randomized_eig, SparseLR
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format, is_square, is_symmetric, check_weights
from sknetwork.utils.preprocessing import connected_components
from typing import Union


class Spectral(Algorithm):
    """Weighted spectral embedding of a graph

        Parameters
        ----------
        embedding_dimension : int, optional
            Dimension of the embedding space
        node_weights : {``'uniform'``, ``'degree'``, array of length n_nodes with positive entries}, optional
            Weights used for the normalization for the laplacian, :math:`W^{-1/2} L W^{-1/2}`
        low_rank_regularization: ``None`` or float (default=0.01)
            Implicitly add edges of given weight between all pairs of nodes.
        energy_scaling: bool (default=True)
            If ``True``, rescales each column of the embedding by dividing it by :math:`\\sqrt{\\lambda_i}`.
            Only valid if ``node_weights == 'degree'``.

        Attributes
        ----------
        embedding_ : array, shape = (n_nodes, embedding_dimension)
            Embedding matrix of the nodes
        eigenvalues_ : array, shape = (embedding_dimension)
            Smallest eigenvalues of the training matrix

        Example
        -------
        >>> from sknetwork.toy_graphs import karate_club_graph
        >>> graph = karate_club_graph()
        >>> spectral = Spectral(embedding_dimension=2)
        >>> spectral.fit(graph)
        Spectral(embedding_dimension=2, node_weights='degree', low_rank_regularization=0.01, energy_scaling=True)
        >>> spectral.embedding_.shape
        (34, 2)

        References
        ----------
        * Weighted Spectral Embedding, T. Bonald
          https://arxiv.org/abs/1809.11115
        * Laplacian Eigenmaps for Dimensionality Reduction and Data Representation, M. Belkin, P. Niyogi
        """

    def __init__(self, embedding_dimension: int = 2, node_weights='degree',
                 low_rank_regularization: Union[None, float] = 0.01, energy_scaling: bool = True):
        self.embedding_dimension = embedding_dimension
        self.node_weights = node_weights
        self.low_rank_regularization = low_rank_regularization
        self.energy_scaling = energy_scaling
        self.embedding_ = None
        self.eigenvalues_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], node_weights=None,
            randomized_decomposition: bool = True) -> 'Spectral':
        """Fits the model from data in adjacency_matrix

        Parameters
        ----------
        adjacency : array-like, shape = (n, n)
              Adjacency matrix of the graph
        randomized_decomposition: bool (default=True)
            whether to use a randomized (and faster) decomposition method or the standard scipy one.
        node_weights : {``'uniform'``, ``'degree'``, array of length n_nodes with positive entries}
              Node weights

        Returns
        -------
        self: :class:`Spectral`
        """

        adjacency = check_format(adjacency)
        n_nodes, m_nodes = adjacency.shape
        if not is_square(adjacency):
            raise ValueError("The adjacency matrix must be a square matrix.")
        if connected_components(adjacency, directed=False)[0] > 1 and self.low_rank_regularization is None:
            if self.energy_scaling:
                raise ValueError('energy_scaling without low-rank regularization'
                                 'is not compatible with a disconnected graph')
            else:
                raise Warning("The graph is not connected and low-rank regularization is set to None."
                              "This can cause errors in the computation of the embedding.")
        if not is_symmetric(adjacency):
            raise ValueError("The adjacency matrix is not symmetric.")
        if self.low_rank_regularization:
            adjacency = SparseLR(adjacency, [(self.low_rank_regularization * np.ones(n_nodes), np.ones(n_nodes))])

        # builds standard laplacian
        degrees = adjacency.dot(np.ones(n_nodes))
        degree_matrix = sparse.diags(degrees, format='csr')
        laplacian = -(adjacency - degree_matrix)

        # applies normalization by node weights
        if node_weights is None:
            node_weights = self.node_weights
        weights = check_weights(node_weights, adjacency, positive_entries=False)

        weight_matrix = sparse.diags(np.sqrt(weights), format='csr')
        weight_matrix.data = 1 / weight_matrix.data
        laplacian = safe_sparse_dot(weight_matrix, safe_sparse_dot(laplacian, weight_matrix))

        # spectral decomposition
        n_components = min(self.embedding_dimension + 1, n_nodes - 1)
        if randomized_decomposition:
            eigenvalues, eigenvectors = randomized_eig(laplacian, n_components, which='SM')
        else:
            eigenvalues, eigenvectors = eigsh(laplacian, n_components, which='SM')

        self.eigenvalues_ = eigenvalues[1:]
        self.embedding_ = np.array(weight_matrix.dot(eigenvectors[:, 1:]))

        if self.energy_scaling and node_weights == 'degree':
            self.embedding_ /= np.sqrt(self.eigenvalues_)
        return self
