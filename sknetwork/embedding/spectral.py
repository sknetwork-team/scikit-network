#!/usr/bin/env python3
# coding: utf-8

"""
Created on Thu Sep 13 2018

Authors:
Thomas Bonald <thomas.bonald@telecom-paristech.fr>
Nathan De Lara <nathan.delara@telecom-paristech.fr>

Spectral embedding by decomposition of the normalized graph Laplacian.
"""

import numpy as np

from sknetwork.embedding.randomized_matrix_factorization import randomized_eig
from scipy import sparse, errstate, sqrt, isinf
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


class SpectralEmbedding:
    """Weighted spectral embedding of a graph

        Attributes
        ----------
        embedding_dimension : int, optional
            Dimension of the embedding space (default=100)
        eigenvalue_normalization : bool, optional
            Whether to normalize the embedding by the pseudo-inverse square roots of laplacian eigenvalues
            (default=True)
        node_weights : {'uniform', 'degree', array of length n_nodes with positive entries}, optional
            Weights used for the normalization for the laplacian, :math:`W^{-1/2} L W^{-1/2}`
        embedding_ : array, shape = (n_nodes, embedding_dimension)
            Embedding matrix of the nodes
        eigenvalues_ : array, shape = (embedding_dimension)
            Smallest eigenvalues of the training matrix

        References
        ----------
        * Weighted Spectral Embedding, T. Bonald
        * Laplacian Eigenmaps for Dimensionality Reduction and Data Representation, M. Belkin, P. Niyogi
        """

    def __init__(self, embedding_dimension: int = 2, node_weights='degree', eigenvalue_normalization: bool = True):
        self.embedding_dimension = embedding_dimension
        self.node_weights = node_weights
        self.eigenvalue_normalization = eigenvalue_normalization
        self.embedding_ = None
        self.eigenvalues_ = None

    def fit(self, adjacency_matrix, node_weights=None, randomized_decomposition: bool = True):
        """Fits the model from data in adjacency_matrix

        Parameters
        ----------
        adjacency_matrix : Scipy csr matrix or numpy ndarray
              Adjacency matrix of the graph
        randomized_decomposition: whether to use a randomized (and faster) decomposition method or
            the standard scipy one.
        node_weights : {'uniform', 'degree', array of length n_nodes with positive entries}
              Node weights
        """

        if type(adjacency_matrix) == sparse.csr_matrix:
            adj_matrix = adjacency_matrix
        elif sparse.isspmatrix(adjacency_matrix) or type(adjacency_matrix) == np.ndarray:
            adj_matrix = sparse.csr_matrix(adjacency_matrix)
        else:
            raise TypeError(
                "The argument must be a NumPy array or a SciPy Sparse matrix.")
        n_nodes, m_nodes = adj_matrix.shape
        if n_nodes != m_nodes:
            raise ValueError("The adjacency matrix must be a square matrix.")
        if csgraph.connected_components(adj_matrix, directed=False)[0] > 1:
            raise ValueError("The graph must be connected.")
        if (adj_matrix != adj_matrix.maximum(adj_matrix.T)).nnz != 0:
            raise ValueError("The adjacency matrix is not symmetric.")

        # builds standard laplacian
        degrees = adj_matrix.dot(np.ones(n_nodes))
        degree_matrix = sparse.diags(degrees, format='csr')
        laplacian = degree_matrix - adj_matrix

        # applies normalization by node weights
        if node_weights is None:
            node_weights = self.node_weights
        if type(node_weights) == str:
            if node_weights == 'uniform':
                weight_matrix = sparse.identity(n_nodes)
            elif node_weights == 'degree':
                with errstate(divide='ignore'):
                    degrees_inv_sqrt = 1.0 / sqrt(degrees)
                degrees_inv_sqrt[isinf(degrees_inv_sqrt)] = 0
                weight_matrix = sparse.diags(degrees_inv_sqrt, format='csr')
            else:
                raise ValueError('Unknown weighting policy. Try \'degree\' or \'uniform\'.')
        else:
            if len(self.node_weights) != n_nodes:
                raise ValueError('node_weights must be an array of length n_nodes.')
            elif min(self.node_weights) < 0:
                raise ValueError('node_weights must be positive.')
            else:
                with errstate(divide='ignore'):
                    weights_inv_sqrt = 1.0 / sqrt(self.node_weights)
                weights_inv_sqrt[isinf(weights_inv_sqrt)] = 0
                weight_matrix = sparse.diags(weights_inv_sqrt, format='csr')

        laplacian = weight_matrix.dot(laplacian.dot(weight_matrix))

        # spectral decomposition
        n_components = min(self.embedding_dimension + 1, n_nodes - 1)
        if randomized_decomposition:
            eigenvalues, eigenvectors = randomized_eig(laplacian, n_components, which='SM')
        else:
            eigenvalues, eigenvectors = eigsh(laplacian, n_components, which='SM')

        self.eigenvalues_ = eigenvalues[1:]

        self.embedding_ = np.array(weight_matrix.dot(eigenvectors[:, 1:]))
        if self.eigenvalue_normalization:
            eigenvalues_inv_sqrt = 1.0 / sqrt(eigenvalues[1:])
            self.embedding_ = eigenvalues_inv_sqrt * self.embedding_

        return self
