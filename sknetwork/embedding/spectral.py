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
from scipy.sparse.linalg import LinearOperator

from sknetwork.basics.structure import is_connected
from sknetwork.linalg import EigSolver, HalkoEig, LanczosEig, auto_solver
from sknetwork.utils.adjacency_formats import set_adjacency
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format


class LaplacianOperator(LinearOperator):
    """
    Regularized Laplacian matrix as a scipy LinearOperator.
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], regularization: float = 0.):
        LinearOperator.__init__(self, dtype=float, shape=adjacency.shape)
        self.regularization = regularization
        self.weights = adjacency.dot(np.ones(adjacency.shape[1]))
        self.laplacian = sparse.diags(self.weights, format='csr') - adjacency

    def _matvec(self, matrix: np.ndarray):
        prod = self.laplacian.dot(matrix)
        prod += self.shape[0] * self.regularization * matrix
        if len(matrix.shape) == 2:
            prod -= self.regularization * np.tile(matrix.sum(axis=0), (self.shape[0], 1))
        else:
            prod -= self.regularization * matrix.sum()

        return prod

    def _transpose(self):
        return self

    def astype(self, dtype: Union[str, np.dtype]):
        """Change dtype of the object.

        Parameters
        ----------
        dtype

        Returns
        -------
        self

        """
        self.dtype = np.dtype(dtype)
        self.laplacian = self.laplacian.astype(self.dtype)
        self.weights = self.weights.astype(self.dtype)

        return self


class NormalizedAdjacencyOperator(LinearOperator):
    """
    Regularized normalized adjacency matrix as a scipy LinearOperator.
    """
    def __init__(self, adjacency: Union[sparse.csr_matrix, np.ndarray], regularization: float = 0.):
        LinearOperator.__init__(self, dtype=float, shape=adjacency.shape)
        self.adjacency = adjacency
        self.regularization = regularization

        n = self.adjacency.shape[0]
        self.sqrt_weights = np.sqrt(self.adjacency.dot(np.ones(n)) + self.regularization * n)

    def _matvec(self, matrix: np.ndarray):
        matrix = (matrix.T / self.sqrt_weights).T
        prod = self.adjacency.dot(matrix)
        if len(matrix.shape) == 2:
            prod += self.regularization * np.tile(matrix.sum(axis=0), (self.shape[0], 1))
        else:
            prod += self.regularization * matrix.sum()
        return (prod.T / self.sqrt_weights).T

    def _transpose(self):
        return self

    def astype(self, dtype: Union[str, np.dtype]):
        """Change dtype of the object.

        Parameters
        ----------
        dtype

        Returns
        -------
        self

        """
        self.dtype = np.dtype(dtype)
        self.adjacency = self.adjacency.astype(self.dtype)
        self.sqrt_weights = self.sqrt_weights.astype(self.dtype)

        return self


class Spectral(Algorithm):
    """
    Spectral embedding of a graph.

    Parameters
    ----------
    embedding_dimension : int, optional
        Dimension of the embedding space
    normalized_laplacian : bool (default = ``True``)
        If ``True``, use the normalized Laplacian, :math:`I - D^{-1/2} A D^{-1/2}`.
    regularization : ``None`` or float (default = ``0.01``)
        Implicitly add edges of given weight between all pairs of nodes.
    relative_regularization : bool (default = ``True``)
        If ``True``, consider the regularization as relative to the total weight of the graph.
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
    col_embedding_ : array, shape = (p, embedding_dimension)
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
                 regularization: Union[None, float] = 0.01, relative_regularization: bool = True,
                 scaling: Union[None, str] = 'multiply', solver: Union[str, EigSolver] = 'auto'):
        self.embedding_dimension = embedding_dimension
        self.normalized_laplacian = normalized_laplacian
        if regularization == 0:
            self.regularization = None
        else:
            self.regularization = regularization
        self.relative_regularization = relative_regularization
        self.scaling = scaling
        if solver == 'halko':
            self.solver: EigSolver = HalkoEig(which='SM')
        elif solver == 'lanczos':
            self.solver: EigSolver = LanczosEig(which='SM')
        else:
            self.solver = solver

        self.embedding_ = None
        self.col_embedding_ = None
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

        adjacency = check_format(adjacency).asfptype()
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

        if (self.regularization is None or self.regularization == 0.) and not is_connected(adjacency):
            warnings.warn(Warning("The graph is not connected and low-rank regularization is not active."
                                  "This can cause errors in the computation of the embedding."))

        if isinstance(self.solver, HalkoEig) and not self.normalized_laplacian:
            raise NotImplementedError('Halko solver is not yet compatible with unormalized Laplacian.'
                                      'Please chose normalized Laplacian or force lanczos solver.')

        weights = adjacency.dot(np.ones(n))
        regularization = self.regularization
        if regularization:
            if self.relative_regularization:
                regularization = regularization * weights.sum() / n**2
            weights += regularization * n

        if self.normalized_laplacian:
            # Finding the largest eigenvectors of the normalized adjacency is easier for the solver than finding the
            # smallest ones of the normalized laplacian.
            normalizing_matrix = sparse.diags(np.sqrt(weights), format='csr')
            normalizing_matrix.data = 1 / normalizing_matrix.data

            if regularization:
                norm_adjacency = NormalizedAdjacencyOperator(adjacency, regularization)
            else:
                norm_adjacency = normalizing_matrix.dot(adjacency.dot(normalizing_matrix))

            self.solver.which = 'LA'
            self.solver.fit(matrix=norm_adjacency, n_components=n_components)
            self.solver.eigenvalues_ = 1 - self.solver.eigenvalues_
            # eigenvalues of the laplacian by increasing order
            index = np.argsort(self.solver.eigenvalues_)
            eigenvalues = self.solver.eigenvalues_[index][1:]
            embedding = self.solver.eigenvectors_[:, index][:, 1:]
            embedding = np.array(normalizing_matrix.dot(embedding))

        else:
            if regularization:
                laplacian = LaplacianOperator(adjacency, regularization)
            else:
                weight_matrix = sparse.diags(weights, format='csr')
                laplacian = weight_matrix - adjacency

            self.solver.which = 'SM'
            self.solver.fit(matrix=laplacian, n_components=n_components)
            eigenvalues = self.solver.eigenvalues_[1:]
            embedding = self.solver.eigenvectors_[:, 1:]

        if self.scaling:
            if self.scaling == 'multiply':
                if self.normalized_laplacian:
                    eigenvalues = np.minimum(eigenvalues, 1)
                    embedding *= np.sqrt(1 - eigenvalues)
                else:
                    embedding *= np.sqrt(np.clip(eigenvalues, a_min=0, a_max=np.max(eigenvalues)))
            elif self.scaling == 'divide':
                inv_eigenvalues = np.zeros_like(eigenvalues)
                index = np.where(eigenvalues > 0)[0]
                inv_eigenvalues[index] = 1 / eigenvalues[index]
                embedding *= np.sqrt(inv_eigenvalues)
            else:
                warnings.warn(Warning("The scaling must be 'multiply' or 'divide'. No scaling done."))

        if n > n1:
            self.embedding_ = embedding[:n1]
            self.col_embedding_ = embedding[n1:]
        else:
            self.embedding_ = embedding
        self.eigenvalues_ = eigenvalues

        return self
