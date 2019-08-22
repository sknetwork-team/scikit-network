#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Sep 13 2018

Authors:
Thomas Bonald <thomas.bonald@telecom-paristech.fr>
Nathan De Lara <nathan.delara@telecom-paristech.fr>
"""

import warnings
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.basics.structure import is_connected
from sknetwork.linalg import safe_sparse_dot, SparseLR, EigSolver, HalkoEig, LanczosEig, auto_solver
from sknetwork.utils.adjacency_formats import set_adjacency
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format


class Spectral(Algorithm):
    """
    Spectral embedding of a graph.

    Parameters
    ----------
    embedding_dimension : int, optional
        Dimension of the embedding space
    normalized_laplacian : bool (default = ``True``)
        If ``True``, use the normalized Laplacian, :math:`I - D^{-1/2} A D^{-1/2}`.
    regularization : ``None`` or float (default= ``0.01``)
        Implicitly add edges of given weight between all pairs of nodes.
    scaling : ``None`` or ``'multiply'`` or ``'divide'`` (default = ``'multiply'``)
        If ```'multiply'``, multiply by the square-root of each eigenvalue.
    solver: ``'auto'``, ``'halko'``, ``'lanczos'`` or :class:`EigSolver`
        Which eigenvalue solver to use.

        * ``'auto'`` call the auto_solver function.
        * ``'halko'``: randomized method, fast but less accurate than ``'lanczos'`` for ill-conditioned matrices.
        * ``'lanczos'``: power-iteration based method.
        * :class:`EigSolver`: custom solver.

    Attributes
    ----------
    embedding_ : array, shape = (n, embedding_dimension)
        Embedding of the nodes.
    coembedding_ : array, shape = (p, embedding_dimension)
        Co-embedding of the feature nodes.
        Only relevant for an asymmetric input matrix or if **force_biadjacency** = ``True``.
    eigenvalues_ : array, shape = (embedding_dimension)
        Eigenvalues in increasing order (first eigenvalue ignored).

    Example
    -------
    >>> from sknetwork.toy_graphs import house
    >>> adjacency = house()
    >>> spectral = Spectral()
    >>> embedding = spectral.fit(adjacency).embedding_
    >>> embedding.shape
    (5, 2)

    References
    ----------
    Belkin, M. & Niyogi, P. (2003). Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
    Neural computation.

    """

    def __init__(self, embedding_dimension: int = 2, normalized_laplacian=True,
                 regularization: Union[None, float] = 0.01, scaling: Union[None, str] = 'multiply',
                 solver: Union[str, EigSolver] = 'auto'):
        self.embedding_dimension = embedding_dimension
        self.normalized_laplacian = normalized_laplacian
        if regularization == 0:
            self.regularization = None
        else:
            self.regularization = regularization
        self.scaling = scaling
        if solver == 'halko':
            self.solver: EigSolver = HalkoEig(which='SM')
        elif solver == 'lanczos':
            self.solver: EigSolver = LanczosEig(which='SM')
        else:
            self.solver = solver

        self.embedding_ = None
        self.coembedding_ = None
        self.eigenvalues_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], force_biadjacency: bool = False) -> 'Spectral':
        """Fits the model from data in adjacency_matrix

        Parameters
        ----------
        adjacency :
              Adjacency or biadjacency matrix of the graph.
        force_biadjacency :
            If ``True``, force the input matrix to be considered as a biadjacency matrix.

        Returns
        -------
        self: :class:`Spectral`
        """

        adjacency = check_format(adjacency)
        n1, n2 = adjacency.shape
        adjacency = set_adjacency(adjacency, force_undirected=True, force_biadjacency=force_biadjacency)
        n = adjacency.shape[0]

        if self.solver == 'auto':
            solver = auto_solver(adjacency.nnz)
            if solver == 'lanczos':
                self.solver: EigSolver = LanczosEig()
            else:
                self.solver: EigSolver = HalkoEig()

        if self.embedding_dimension > n - 2:
            warnings.warn(Warning("The dimension of the embedding must be less than the number of nodes - 1."))
            n_components = n - 2
        else:
            n_components = self.embedding_dimension + 1

        if self.regularization is None and not is_connected(adjacency):
            warnings.warn(Warning("The graph is not connected and low-rank regularization is not active."
                                  "This can cause errors in the computation of the embedding."))

        if self.regularization:
            adjacency = SparseLR(adjacency, [(self.regularization * np.ones(n), np.ones(n))])

        weights = adjacency.dot(np.ones(n))

        if self.normalized_laplacian:
            normalizing_matrix = sparse.diags(np.sqrt(weights), format='csr')
            normalizing_matrix.data = 1 / normalizing_matrix.data
            norm_adjacency = safe_sparse_dot(normalizing_matrix, safe_sparse_dot(adjacency, normalizing_matrix))
            self.solver.which = 'LA'
            self.solver.fit(norm_adjacency, n_components)
            # sort the eigenvalues by decreasing order
            self.solver.eigenvalues_ = self.solver.eigenvalues_[::-1]
            self.solver.eigenvectors_ = self.solver.eigenvectors_[:, ::-1]

            # eigenvalues of the laplacian by increasing order
            self.eigenvalues_ = 1 - self.solver.eigenvalues_[1:]
            self.embedding_ = self.solver.eigenvectors_[:, 1:]
            self.embedding_ = np.array(normalizing_matrix.dot(self.embedding_))
        else:
            weight_matrix = sparse.diags(weights, format='csr')
            laplacian = -(adjacency - weight_matrix)
            self.solver.which = 'SM'
            self.solver.fit(laplacian, n_components)
            self.eigenvalues_ = self.solver.eigenvalues_[1:]
            self.embedding_ = self.solver.eigenvectors_[:, 1:]

        if self.scaling:
            if self.scaling == 'multiply':
                if self.normalized_laplacian:
                    self.eigenvalues_ = np.minimum(self.eigenvalues_, 1)
                    self.embedding_ *= np.sqrt(1 - self.eigenvalues_)
                else:
                    self.embedding_ *= np.sqrt(abs(self.eigenvalues_))
            elif self.scaling == 'divide':
                inv_eigenvalues = np.zeros(len(self.eigenvalues_))
                index = np.where(self.eigenvalues_ > 0)[0]
                inv_eigenvalues[index] = 1 / self.eigenvalues_[index]
                self.embedding_ *= np.sqrt(inv_eigenvalues)
            else:
                warnings.warn(Warning("The scaling must be 'multiply' or 'divide'. No scaling done."))
        if n > n1:
            self.coembedding_ = self.embedding_[n1:]
            self.embedding_ = self.embedding_[:n1]

        return self
