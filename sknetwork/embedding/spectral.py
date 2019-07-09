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
from sknetwork.linalg import safe_sparse_dot, SparseLR, EigSolver, HalkoEig, LanczosEig
from sknetwork.utils.adjacency_formats import bipartite2undirected
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format, check_weights
from sknetwork.utils.preprocessing import is_connected
from typing import Union


class Spectral(Algorithm):
    """Weighted spectral embedding of a graph.

        Parameters
        ----------
        embedding_dimension : int, optional
            Dimension of the embedding space
        node_weights : {``'uniform'``, ``'degree'``, array of length n_nodes with positive entries}, optional
            Weights used for the normalization for the laplacian, :math:`W^{-1/2} L W^{-1/2}`
        regularization: ``None`` or float (default=0.01)
            Implicitly add edges of given weight between all pairs of nodes.
        energy_scaling: bool (default=True)
            If ``True``, rescales each column of the embedding by dividing it by :math:`\\sqrt{\\lambda_i}`.
            Only valid if ``node_weights == 'degree'``.
        force_biadjacency: bool (default=False)
            Only relevant for symmetric inputs. Force the algorithm to treat the adjacency as a biadjacency
            as it would do for asymmetric inputs.
        solver: 'halko', 'lanczos' or EigSolver object
            Which eigenvalue solver to use
            * 'halko': randomized method, fast but less accurate than 'lanczos' for ill-conditioned matrices
            * 'lanczos': power-iteration based method
            * custom: the user must provide an EigSolver object.

        Attributes
        ----------
        embedding_ : array, shape = (n_nodes, embedding_dimension)
            Embedding matrix of the nodes
        features_ : array, shape = (m_nodes, embedding_dimension)
            Only relevant for asymmetric inputs or if ``force_biadjacency==True``.
        eigenvalues_ : array, shape = (embedding_dimension)
            Smallest eigenvalues of the training matrix

        Example
        -------
        >>> from sknetwork.toy_graphs import karate_club_graph
        >>> graph = karate_club_graph()
        >>> spectral = Spectral(embedding_dimension=2)
        >>> spectral.fit(graph)
        Spectral(embedding_dimension=2, node_weights='degree', regularization=0.01, energy_scaling=True,\
 force_biadjacency=False, solver=HalkoEig(which='SM', n_oversamples=10, n_iter='auto',\
 power_iteration_normalizer='auto', one_pass=False))
        >>> spectral.embedding_.shape
        (34, 2)

        References
        ----------
        * Weighted Spectral Embedding, T. Bonald
          https://arxiv.org/abs/1809.11115
        * Laplacian Eigenmaps for Dimensionality Reduction and Data Representation, M. Belkin, P. Niyogi
        """

    def __init__(self, embedding_dimension: int = 2, node_weights='degree',
                 regularization: Union[None, float] = 0.01, energy_scaling: bool = True,
                 force_biadjacency: bool = False, solver: Union[str, EigSolver] = 'halko'):
        self.embedding_dimension = embedding_dimension
        self.node_weights = node_weights
        if regularization == 0:
            self.regularization = None
        else:
            self.regularization = regularization
        self.energy_scaling = energy_scaling
        self.force_biadjacency = force_biadjacency
        if solver == 'halko':
            self.solver: EigSolver = HalkoEig(which='SM')
        elif solver == 'lanczos':
            self.solver: EigSolver = LanczosEig(which='SM')
        else:
            self.solver: EigSolver = solver

        self.embedding_ = None
        self.features_ = None
        self.eigenvalues_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Spectral':
        """Fits the model from data in adjacency_matrix

        Parameters
        ----------
        adjacency : array-like, shape = (n, n)
              Adjacency matrix of the graph

        Returns
        -------
        self: :class:`Spectral`
        """

        adjacency = check_format(adjacency)
        n_nodes, m_nodes = adjacency.shape
        if self.regularization is None and not is_connected(adjacency):
            if self.energy_scaling:
                raise ValueError('energy_scaling without low-rank regularization'
                                 'is not compatible with a disconnected graph')
            else:
                raise Warning("The graph is not connected and low-rank regularization is not active."
                              "This can cause errors in the computation of the embedding.")
        if self.regularization:
            adjacency = SparseLR(adjacency, [(self.regularization * np.ones(n_nodes), np.ones(m_nodes))])
        if m_nodes != n_nodes or self.force_biadjacency:
            adjacency = bipartite2undirected(adjacency)

        # builds standard laplacian
        degrees = adjacency.dot(np.ones(adjacency.shape[1]))
        degree_matrix = sparse.diags(degrees, format='csr')
        laplacian = -(adjacency - degree_matrix)

        # applies normalization by node weights
        weights = check_weights(self.node_weights, adjacency, positive_entries=False)
        weight_matrix = sparse.diags(np.sqrt(weights), format='csr')
        weight_matrix.data = 1 / weight_matrix.data

        laplacian = safe_sparse_dot(weight_matrix, safe_sparse_dot(laplacian, weight_matrix))

        # spectral decomposition
        n_components = min(self.embedding_dimension + 1, min(n_nodes, m_nodes))
        self.solver.fit(laplacian, n_components)

        self.eigenvalues_ = self.solver.eigenvalues_[1:]
        self.embedding_ = np.array(weight_matrix.dot(self.solver.eigenvectors_[:, 1:]))

        if self.energy_scaling and self.node_weights == 'degree':
            self.embedding_ /= np.sqrt(self.eigenvalues_)

        if self.embedding_.shape[0] > n_nodes:
            self.features_ = self.embedding_[n_nodes:]
            self.embedding_ = self.embedding_[:n_nodes]
            
        return self
