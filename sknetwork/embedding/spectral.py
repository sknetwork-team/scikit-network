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
from sknetwork.embedding.base import BaseEmbedding
from sknetwork.linalg import EigSolver, HalkoEig, LanczosEig, auto_solver, diag_pinv
from sknetwork.utils.checks import check_format, is_square, is_symmetric


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


class Spectral(BaseEmbedding):
    """
    Spectral embedding of a graph.

    Solves the eigenvalue problem :math:`LU = U\\Lambda`, where :math:`L` is the graph Laplacian.

    The embedding is :math:`X = U \\phi(\\Lambda)` where :math:`\\phi(\\Lambda)` is a diagonal scaling matrix.

    Parameters
    ----------
    embedding_dimension : int (default = 2)
        Dimension of the embedding space
    normalized_laplacian : bool (default = ``True``)

        * If ``True``, use the normalized Laplacian, :math:`L = I - D^{-1/2} A D^{-1/2}`.
        * If ``False``, use the regular Laplacian, :math:`L = D - A`.
    regularization : ``None`` or float (default = ``0.01``)
        Implicitly add edges of given weight between all pairs of nodes.
    relative_regularization : bool (default = ``True``)
        If ``True``, consider the regularization as relative to the total weight of the graph.
    scaling:  ``None`` or ``'multiply'`` or ``'divide'`` or ``'barycenter'`` (default = ``'multiply'``)

        * ``None``: :math:`\\phi(\\Lambda) = I`,
        * ``'multiply'`` : :math:`\\phi(\\Lambda) = \\sqrt{\\Lambda}`,
        * ``'divide'``  : :math:`\\phi(\\Lambda)= (\\sqrt{1 - \\Lambda})^{-1}`.
    solver: ``'auto'``, ``'halko'``, ``'lanczos'`` or :class:`EigSolver` (default = ``'auto'``)
        Which eigenvalue solver to use.

        * ``'auto'`` call the auto_solver function.
        * ``'halko'``: randomized method, fast but less accurate than ``'lanczos'`` for ill-conditioned matrices.
        * ``'lanczos'``: power-iteration based method.
        * :class:`EigSolver`: custom solver.
    tol: float (default = 1e-10)
        Skip eigenvectors of the normalized Laplacian with eigenvalues larger than 1 - tol.

    Attributes
    ----------
    embedding_ : array, shape = (n, embedding_dimension)
        Embedding of the nodes.
    eigenvalues_ : array, shape = (embedding_dimension)
        Eigenvalues in increasing order (first eigenvalue ignored).
    regularization_ : ``None`` or float
        Regularization factor added to all pairs of nodes.

    Example
    -------
    >>> from sknetwork.data import house
    >>> adjacency = house()
    >>> spectral = Spectral()
    >>> embedding = spectral.fit_transform(adjacency)
    >>> embedding.shape
    (5, 2)

    References
    ----------
    Belkin, M. & Niyogi, P. (2003). Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
    Neural computation.

    """

    def __init__(self, embedding_dimension: int = 2, normalized_laplacian=True,
                 regularization: Union[None, float] = 0.01, relative_regularization: bool = True,
                 scaling: Union[None, str] = 'multiply', solver: Union[str, EigSolver] = 'auto', tol: float = 1e-10):
        super(Spectral, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.normalized_laplacian = normalized_laplacian

        if regularization == 0:
            self.regularization = None
        else:
            self.regularization = regularization
        self.relative_regularization = relative_regularization

        self.scaling = scaling
        if scaling == 'multiply' and not normalized_laplacian:
            self.scaling = None
            warnings.warn(Warning("The scaling 'multiply' is valid only with ``normalized_laplacian = 'True'``. "
                                  "It will be ignored."))

        if solver == 'halko':
            self.solver: EigSolver = HalkoEig(which='SM')
        elif solver == 'lanczos':
            self.solver: EigSolver = LanczosEig(which='SM')
        else:
            self.solver = solver

        self.tol = tol

        self.eigenvalues_ = None
        self.regularization_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Spectral':
        """Fits the model from data in adjacency.

        Parameters
        ----------
        adjacency :
              Adjacency matrix of the graph (symmetric matrix).

        Returns
        -------
        self: :class:`Spectral`
        """

        adjacency = check_format(adjacency).asfptype()

        if not is_square(adjacency):
            raise ValueError('The adjacency matrix is not square. See BiSpectral.')

        if not is_symmetric(adjacency):
            raise ValueError('The adjacency matrix is not symmetric.'
                             'Either convert it to a symmetric matrix or use BiSpectral.')

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
            raise NotImplementedError("Halko solver is not yet compatible with regular Laplacian."
                                      "Call 'fit' with 'normalized_laplacian' = True or force lanczos solver.")

        weights = adjacency.dot(np.ones(n))
        regularization = self.regularization
        if regularization:
            if self.relative_regularization:
                regularization = regularization * weights.sum() / n ** 2
            weights += regularization * n

        if self.normalized_laplacian:
            # Finding the largest eigenvalues of the normalized adjacency is easier for the solver than finding the
            # smallest eigenvalues of the normalized laplacian.
            normalizing_matrix = diag_pinv(np.sqrt(weights))

            if regularization:
                norm_adjacency = NormalizedAdjacencyOperator(adjacency, regularization)
            else:
                norm_adjacency = normalizing_matrix.dot(adjacency.dot(normalizing_matrix))

            self.solver.which = 'LA'
            self.solver.fit(matrix=norm_adjacency, n_components=n_components)
            eigenvalues = 1 - self.solver.eigenvalues_
            # eigenvalues of the Laplacian in increasing order
            index = np.argsort(eigenvalues)
            # skip first eigenvalue
            eigenvalues = eigenvalues[index][1:]
            # keep only positive eigenvectors of the normalized adjacency matrix
            eigenvectors = self.solver.eigenvectors_[:, index][:, 1:] * (eigenvalues < 1 - self.tol)
            embedding = np.array(normalizing_matrix.dot(eigenvectors))

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
                eigenvalues = np.minimum(eigenvalues, 1)
                embedding *= np.sqrt(1 - eigenvalues)
            elif self.scaling == 'divide':
                inv_eigenvalues = np.zeros_like(eigenvalues)
                index = np.where(eigenvalues > 0)[0]
                inv_eigenvalues[index] = 1 / eigenvalues[index]
                embedding *= np.sqrt(inv_eigenvalues)
            else:
                warnings.warn(Warning("The scaling must be 'multiply' or 'divide'. No scaling done."))

        self.embedding_ = embedding
        self.eigenvalues_ = eigenvalues
        self.regularization_ = regularization

        return self

    def predict(self, adjacency_vector: np.ndarray) -> np.ndarray:
        """Predicts the embedding of a new node, defined by its adjacency vector.

        Parameters
        ----------
        adjacency_vector : array, shape (n,)
              Adjacency vector of a node.

        Returns
        -------
        embedding_vector : array, shape (embedding_dimension,)
            Embedding of the node.
        """
        embedding = self.embedding_
        eigenvalues = self.eigenvalues_

        if embedding is None:
            raise ValueError("This instance of Spectral embedding is not fitted yet."
                             " Call 'fit' with appropriate arguments before using this method.")
        else:
            n = embedding.shape[0]

        if adjacency_vector.shape[0] != n:
            raise ValueError('The adjacency vector must be of length equal to the number of nodes.')
        elif not np.all(adjacency_vector >= 0):
            raise ValueError('The adjacency vector must be non-negative.')

        # regularization
        reg_adjacency_vector = adjacency_vector.astype(float)
        if self.regularization_:
            reg_adjacency_vector += self.regularization_

        # projection in the embedding space
        if self.normalized_laplacian:
            embedding_vector = np.zeros(self.embedding_dimension)
            index = np.where(eigenvalues < 1 - self.tol)[0]
            embedding_vector[index] = embedding[:, index].T.dot(reg_adjacency_vector) / np.sum(reg_adjacency_vector)
            embedding_vector[index] /= 1 - eigenvalues[index]
        else:
            raise ValueError("The predict method is not available for the spectral embedding based on the Laplacian."
                             " Call 'fit' with 'normalized_laplacian' = True.")

        return embedding_vector
