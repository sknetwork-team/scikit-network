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
from sknetwork.linalg import safe_sparse_dot, SparseLR, EigSolver, HalkoEig, LanczosEig, auto_solver
from sknetwork.utils.adjacency_formats import bipartite2undirected
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format, is_symmetric
from sknetwork.utils.preprocessing import is_connected
from typing import Union


class Spectral(Algorithm):
    """
    Spectral embedding of a graph.

    Parameters
    ----------
    embedding_dimension : int, optional
        Dimension of the embedding space
    normalized_laplacian : bool (default = True)
        If True, uses the normalized Laplacian, :math:`I - D^{-1/2} A D^{-1/2}`
    regularization : ``None`` or float (default=0.01)
        Implicitly add edges of given weight between all pairs of nodes.
    energy_scaling : bool (default=True)
        If ``True``, rescales each column of the embedding by dividing it by the square-root of the corresponding
        eigenvalue. Only valid if ``node_weights == 'degree'``.
    force_biadjacency : bool (default=False)
        Forces the input matrix to be considered as a biadjacency matrix. Only relevant for a symmetric input matrix.
    solver: 'auto', 'halko', 'lanczos' or EigSolver object
        Which eigenvalue solver to use

        * ``'auto'`` calls the auto_solver function
        * ``'halko'``: randomized method, fast but less accurate than 'lanczos' for ill-conditioned matrices
        * ``'lanczos'``: power-iteration based method
        * ``custom``: the user must provide an EigSolver object.

    Attributes
    ----------
    embedding_ : array, shape = (n, embedding_dimension)
        Embedding of the nodes
    coembedding_ : array, shape = (p, embedding_dimension)
        Co-embedding of the feature nodes
        Only relevant for an asymmetric input matrix or if ``force_biadjacency==True``.
    eigenvalues_ : array, shape = (embedding_dimension)
        Smallest eigenvalues

    Example
    -------
    >>> from sknetwork.toy_graphs import karate_club
    >>> adjacency = karate_club()
    >>> spectral = Spectral(embedding_dimension=2)
    >>> embedding = spectral.fit(adjacency).embedding_
    >>> embedding.shape
    (34, 2)

    References
    ----------
    Belkin, M. & Niyogi, P. (2003). Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
    Neural computation.

    """

    def __init__(self, embedding_dimension: int = 2, normalized_laplacian=True,
                 regularization: Union[None, float] = 0.01, energy_scaling: bool = True,
                 force_biadjacency: bool = False, solver: Union[str, EigSolver] = 'auto'):
        self.embedding_dimension = embedding_dimension
        self.normalized_laplacian = normalized_laplacian
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
            self.solver = solver

        self.embedding_ = None
        self.coembedding_ = None
        self.eigenvalues_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Spectral':
        """Fits the model from data in adjacency_matrix

        Parameters
        ----------
        adjacency : array-like, shape = (n, p)
              Adjacency or biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`Spectral`
        """

        adjacency = check_format(adjacency)
        n, p = adjacency.shape

        if p != n or not is_symmetric(adjacency) or self.force_biadjacency:
            adjacency = bipartite2undirected(adjacency)

        n_ = adjacency.shape[0]

        if self.solver == 'auto':
            solver = auto_solver(adjacency.nnz)
            if solver == 'lanczos':
                self.solver: EigSolver = LanczosEig(which='SM')
            else:
                self.solver: EigSolver = HalkoEig(which='SM')

        if self.regularization is None and not is_connected(adjacency):
            if self.energy_scaling:
                raise ValueError("The graph is not connected and low-rank regularization is not active."
                                 "This can cause errors in the computation of the embedding."
                                 "The parameter energy_scaling must be set to False")
            else:
                raise Warning("The graph is not connected and low-rank regularization is not active."
                              "This can cause errors in the computation of the embedding.")

        if self.regularization:
            adjacency = SparseLR(adjacency, [(self.regularization * np.ones(n_), np.ones(n_))])

        # builds standard Laplacian
        degrees = adjacency.dot(np.ones(adjacency.shape[1]))
        degree_matrix = sparse.diags(degrees, format='csr')
        laplacian = -(adjacency - degree_matrix)

        # applies normalization of the Laplacian
        if self.normalized_laplacian:
            inv_sqrt_degree_matrix = sparse.diags(np.sqrt(degrees), format='csr')
            inv_sqrt_degree_matrix.data = 1 / inv_sqrt_degree_matrix.data
            laplacian = safe_sparse_dot(inv_sqrt_degree_matrix, safe_sparse_dot(laplacian, inv_sqrt_degree_matrix))

        # spectral decomposition
        n_components = min(self.embedding_dimension + 1, n_)
        self.solver.fit(laplacian, n_components)

        self.eigenvalues_ = self.solver.eigenvalues_[1:]
        self.embedding_ = self.solver.eigenvectors_[:, 1:]
        if self.normalized_laplacian:
            self.embedding_ = np.array(inv_sqrt_degree_matrix.dot(self.embedding_))

        if self.energy_scaling and self.normalized_laplacian:
            self.embedding_ /= np.sqrt(self.eigenvalues_)

        if n_ > n:
            self.coembedding_ = self.embedding_[n:]
            self.embedding_ = self.embedding_[:n]

        return self
