#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Sep 13 2018

Authors:
Thomas Bonald <thomas.bonald@telecom-paristech.fr>
Nathan De Lara <nathan.delara@telecom-paristech.fr>
"""

import numpy as np

from sknetwork.embedding.randomized_matrix_factorization import randomized_eig
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from typing import Union


class SpectralEmbedding:
    """Weighted spectral embedding of a graph

        Parameters
        ----------
        embedding_dimension : int, optional
            Dimension of the embedding space
        node_weights : {``'uniform'``, ``'degree'``, array of length n_nodes with positive entries}, optional
            Weights used for the normalization for the laplacian, :math:`W^{-1/2} L W^{-1/2}`

        Attributes
        ----------
        embedding_ : array, shape = (n_nodes, embedding_dimension)
            Embedding matrix of the nodes
        eigenvalues_ : array, shape = (embedding_dimension)
            Smallest eigenvalues of the training matrix

        References
        ----------
        * Weighted Spectral Embedding, T. Bonald
        * Laplacian Eigenmaps for Dimensionality Reduction and Data Representation, M. Belkin, P. Niyogi
        """

    def __init__(self, embedding_dimension: int = 2, node_weights='degree'):
        self.embedding_dimension = embedding_dimension
        self.node_weights = node_weights
        self.embedding_ = None
        self.eigenvalues_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], node_weights=None,
            randomized_decomposition: bool = True):
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
        self: :class:`SpectralEmbedding`
        """

        if type(adjacency) == sparse.csr_matrix:
            adjacency = adjacency
        elif sparse.isspmatrix(adjacency) or type(adjacency) == np.ndarray:
            adjacency = sparse.csr_matrix(adjacency)
        else:
            raise TypeError(
                "The argument must be a NumPy array or a SciPy Sparse matrix.")
        n_nodes, m_nodes = adjacency.shape
        if n_nodes != m_nodes:
            raise ValueError("The adjacency matrix must be a square matrix.")
        if csgraph.connected_components(adjacency, directed=False)[0] > 1:
            raise ValueError("The graph must be connected.")
        if (adjacency != adjacency.maximum(adjacency.T)).nnz != 0:
            raise ValueError("The adjacency matrix is not symmetric.")

        # builds standard laplacian
        degrees = adjacency.dot(np.ones(n_nodes))
        degree_matrix = sparse.diags(degrees, format='csr')
        laplacian = degree_matrix - adjacency

        # applies normalization by node weights
        if node_weights is None:
            node_weights = self.node_weights
        if type(node_weights) == str:
            if node_weights == 'uniform':
                weights = np.ones(n_nodes)
            elif node_weights == 'degree':
                weights = degrees
            else:
                raise ValueError('Unknown weighting policy. Try \'degree\' or \'uniform\'.')
        else:
            if len(self.node_weights) != n_nodes:
                raise ValueError('node_weights must be an array of length n_nodes.')
            elif min(self.node_weights) < 0:
                raise ValueError('node_weights must be positive.')
            else:
                weights = self.node_weights

        weight_matrix = sparse.diags(np.sqrt(weights), format='csr')
        weight_matrix.data = 1 / weight_matrix.data
        laplacian = weight_matrix.dot(laplacian.dot(weight_matrix))

        # spectral decomposition
        n_components = min(self.embedding_dimension + 1, n_nodes - 1)
        if randomized_decomposition:
            eigenvalues, eigenvectors = randomized_eig(laplacian, n_components, which='SM')
        else:
            eigenvalues, eigenvectors = eigsh(laplacian, n_components, which='SM')

        self.eigenvalues_ = eigenvalues[1:]
        self.embedding_ = np.array(weight_matrix.dot(eigenvectors[:, 1:]))
        return self
