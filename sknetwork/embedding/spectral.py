#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Sep 13 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding.base import BaseEmbedding, BaseBiEmbedding
from sknetwork.linalg import EigSolver, HalkoEig, LanczosEig, auto_solver, diag_pinv, normalize, LaplacianOperator, \
    NormalizedAdjacencyOperator, RegularizedAdjacency
from sknetwork.utils.check import check_format, check_square, check_symmetry, check_adjacency_vector, \
    check_nonnegative, check_n_components, check_scaling
from sknetwork.utils.format import bipartite2undirected


def set_solver(solver: str, adjacency):
    """Eigenvalue solver based on keyword"""
    if solver == 'auto':
        solver: str = auto_solver(adjacency.nnz)
    if solver == 'lanczos':
        solver: EigSolver = LanczosEig()
    else:  # pragma: no cover
        solver: EigSolver = HalkoEig()
    return solver


class LaplacianEmbedding(BaseEmbedding):
    """Spectral embedding of graphs, based the spectral decomposition of the Laplacian matrix :math:`L = D - A`.
    Eigenvectors are considered in increasing order of eigenvalues, skipping the first eigenvector.

    * Graphs

    Parameters
    ----------
    n_components : int (default = 2)
        Dimension of the embedding space.
    regularization : ``None`` or float (default = ``0.01``)
        Add edges of given weight between all pairs of nodes.
    relative_regularization : bool (default = ``True``)
        If ``True``, consider the regularization as relative to the total weight of the graph.
    scaling : float (non-negative, default = ``0.5``)
        Scaling factor :math:`\\alpha` so that each component is divided by
        :math:`\\lambda^\\alpha`, with :math:`\\lambda` the corresponding eigenvalue of
        the Laplacian matrix :math:`L`. Require regularization if positive and the graph is not connected.
        The default value :math:`\\alpha=\\frac 1 2` equalizes the energy levels of
        the corresponding mechanical system.
    normalized : bool (default = ``True``)
        If ``True``, normalize the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.
    solver : ``'auto'``, ``'halko'``, ``'lanczos'`` (default = ``'auto'``)
        Which eigenvalue solver to use.

        * ``'auto'`` call the auto_solver function.
        * ``'halko'``: randomized method, fast but less accurate than ``'lanczos'`` for ill-conditioned matrices.
        * ``'lanczos'``: power-iteration based method.

    Attributes
    ----------
    embedding_ : array, shape = (n, n_components)
        Embedding of the nodes.
    eigenvalues_ : array, shape = (n_components)
        Eigenvalues in increasing order (first eigenvalue ignored).
    eigenvectors_ : array, shape = (n, n_components)
        Corresponding eigenvectors.
    regularization_ : ``None`` or float
        Regularization factor added to all pairs of nodes.

    Example
    -------
    >>> from sknetwork.embedding import LaplacianEmbedding
    >>> from sknetwork.data import karate_club
    >>> laplacian = LaplacianEmbedding()
    >>> adjacency = karate_club()
    >>> embedding = laplacian.fit_transform(adjacency)
    >>> embedding.shape
    (34, 2)

    References
    ----------
    Belkin, M. & Niyogi, P. (2003). Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
    Neural computation.
    """
    def __init__(self, n_components: int = 2, regularization: Union[None, float] = 0.01,
                 relative_regularization: bool = True, scaling: float = 0.5,
                 normalized: bool = True, solver: str = 'auto'):
        super(LaplacianEmbedding, self).__init__()

        self.n_components = n_components
        self.regularization = None if regularization == 0 else regularization
        self.relative_regularization = relative_regularization
        self.scaling = scaling
        self.normalized = normalized
        self.solver = solver

        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.regularization_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'LaplacianEmbedding':
        """Compute the graph embedding.

        Parameters
        ----------
        adjacency :
              Adjacency matrix of the graph (symmetric matrix).

        Returns
        -------
        self: :class:`LaplacianEmbedding`
        """
        adjacency = check_format(adjacency).asfptype()
        check_square(adjacency)
        check_symmetry(adjacency)
        n = adjacency.shape[0]

        regularize: bool = not (self.regularization is None or self.regularization == 0.)
        check_scaling(self.scaling, adjacency, regularize)

        if regularize:
            solver: EigSolver = LanczosEig()
        else:
            solver = set_solver(self.solver, adjacency)
        n_components = 1 + check_n_components(self.n_components, n-2)

        weights = adjacency.dot(np.ones(n))
        regularization = self.regularization
        if regularization:
            if self.relative_regularization:
                regularization = regularization * weights.sum() / n ** 2
            weights += regularization * n
            laplacian = LaplacianOperator(adjacency, regularization)
        else:
            weight_diag = sparse.diags(weights, format='csr')
            laplacian = weight_diag - adjacency

        solver.which = 'SM'
        solver.fit(matrix=laplacian, n_components=n_components)
        eigenvalues = solver.eigenvalues_[1:]
        eigenvectors = solver.eigenvectors_[:, 1:]

        embedding = eigenvectors.copy()

        if self.scaling:
            eigenvalues_inv_diag = diag_pinv(eigenvalues ** self.scaling)
            embedding = eigenvalues_inv_diag.dot(embedding.T).T

        if self.normalized:
            embedding = normalize(embedding, p=2)

        self.embedding_ = embedding
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        self.regularization_ = regularization

        return self


class Spectral(BaseEmbedding):
    """Spectral embedding of graphs, based the spectral decomposition of the transition matrix
    :math:`P = D^{-1}A`.
    Eigenvectors are considered in decreasing order of eigenvalues, skipping the first eigenvector.

    * Graphs

    See :class:`BiSpectral` for digraphs and bigraphs.

    Parameters
    ----------
    n_components : int (default = ``2``)
        Dimension of the embedding space.
    regularization : ``None`` or float (default = ``0.01``)
        Add edges of given weight between all pairs of nodes.
    relative_regularization : bool (default = ``True``)
        If ``True``, consider the regularization as relative to the total weight of the graph.
    scaling : float (non-negative, default = ``0.5``)
        Scaling factor :math:`\\alpha` so that each component is divided by
        :math:`(1 - \\lambda)^\\alpha`, with :math:`\\lambda` the corresponding eigenvalue of
        the transition matrix :math:`P`. Require regularization if positive and the graph is not connected.
        The default value :math:`\\alpha=\\frac 1 2` equalizes the energy levels of
        the corresponding mechanical system.
    normalized : bool (default = ``True``)
        If ``True``, normalize the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.
    solver : ``'auto'``, ``'halko'``, ``'lanczos'`` (default = ``'auto'``)
        Which eigenvalue solver to use.

        * ``'auto'`` call the auto_solver function.
        * ``'halko'``: randomized method, fast but less accurate than ``'lanczos'`` for ill-conditioned matrices.
        * ``'lanczos'``: power-iteration based method.

    Attributes
    ----------
    embedding_ : array, shape = (n, n_components)
        Embedding of the nodes.
    eigenvalues_ : array, shape = (n_components)
        Eigenvalues in decreasing order (first eigenvalue ignored).
    eigenvectors_ : array, shape = (n, n_components)
        Corresponding eigenvectors.
    regularization_ : ``None`` or float
        Regularization factor added to all pairs of nodes.

    Example
    -------
    >>> from sknetwork.embedding import Spectral
    >>> from sknetwork.data import karate_club
    >>> spectral = Spectral()
    >>> adjacency = karate_club()
    >>> embedding = spectral.fit_transform(adjacency)
    >>> embedding.shape
    (34, 2)

    References
    ----------
    Belkin, M. & Niyogi, P. (2003). Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
    Neural computation.
    """
    def __init__(self, n_components: int = 2, regularization: Union[None, float] = 0.01,
                 relative_regularization: bool = True, scaling: float = 0.5,
                 normalized: bool = True, solver: str = 'auto'):
        super(Spectral, self).__init__()

        self.n_components = n_components
        self.regularization = None if regularization == 0 else regularization
        self.relative_regularization = relative_regularization
        self.scaling = scaling
        self.normalized = normalized
        self.solver = solver

        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.regularization_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'Spectral':
        """Compute the graph embedding.

        Parameters
        ----------
        adjacency :
              Adjacency matrix of the graph (symmetric matrix).

        Returns
        -------
        self: :class:`Spectral`
        """
        adjacency = check_format(adjacency).asfptype()
        check_square(adjacency)
        check_symmetry(adjacency)
        n = adjacency.shape[0]

        solver = set_solver(self.solver, adjacency)
        n_components = 1 + check_n_components(self.n_components, n-2)

        regularize: bool = not (self.regularization is None or self.regularization == 0.)
        check_scaling(self.scaling, adjacency, regularize)

        weights = adjacency.dot(np.ones(n))
        regularization = self.regularization
        if regularization:
            if self.relative_regularization:
                regularization = regularization * weights.sum() / n ** 2
            weights += regularization * n

        # Spectral decomposition of the normalized adjacency matrix
        weights_inv_sqrt_diag = diag_pinv(np.sqrt(weights))

        if regularization:
            norm_adjacency = NormalizedAdjacencyOperator(adjacency, regularization)
        else:
            norm_adjacency = weights_inv_sqrt_diag.dot(adjacency.dot(weights_inv_sqrt_diag))

        solver.which = 'LA'
        solver.fit(matrix=norm_adjacency, n_components=n_components)
        eigenvalues = solver.eigenvalues_
        index = np.argsort(-eigenvalues)[1:]  # skip first eigenvalue
        eigenvalues = eigenvalues[index]
        eigenvectors = weights_inv_sqrt_diag.dot(solver.eigenvectors_[:, index])

        embedding = eigenvectors.copy()

        if self.scaling:
            eigenvalues_inv_diag = diag_pinv((1 - eigenvalues) ** self.scaling)
            embedding = eigenvalues_inv_diag.dot(embedding.T).T

        if self.normalized:
            embedding = normalize(embedding, p=2)

        self.embedding_ = embedding
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        self.regularization_ = regularization

        return self

    def predict(self, adjacency_vectors: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Predict the embedding of new nodes, defined by their adjacency vectors.

        Parameters
        ----------
        adjacency_vectors :
            Adjacency vectors of nodes.
            Array of shape (n_col,) (single vector) or (n_vectors, n_col)

        Returns
        -------
        embedding_vectors : np.ndarray
            Embedding of the nodes.
        """
        self._check_fitted()
        eigenvectors = self.eigenvectors_
        eigenvalues = self.eigenvalues_
        n = eigenvectors.shape[0]

        adjacency_vectors = check_adjacency_vector(adjacency_vectors, n)
        check_nonnegative(adjacency_vectors)

        # regularization
        if self.regularization_:
            adjacency_vectors = RegularizedAdjacency(adjacency_vectors, self.regularization_)

        # projection in the embedding space
        averaging = normalize(adjacency_vectors, p=1)
        embedding_vectors = averaging.dot(eigenvectors)
        embedding_vectors = diag_pinv(eigenvalues).dot(embedding_vectors.T).T

        if self.scaling:
            eigenvalues_inv_diag = diag_pinv((1 - eigenvalues) ** self.scaling)
            embedding_vectors = eigenvalues_inv_diag.dot(embedding_vectors.T).T

        if self.normalized:
            embedding_vectors = normalize(embedding_vectors, p=2)

        if embedding_vectors.shape[0] == 1:
            embedding_vectors = embedding_vectors.ravel()

        return embedding_vectors


class BiSpectral(Spectral, BaseBiEmbedding):
    """Spectral embedding of bipartite and directed graphs, based the spectral embedding
    of the corresponding undirected graph, with adjacency matrix:

        :math:`A  = \\begin{bmatrix} 0 & B \\\\ B^T & 0 \\end{bmatrix}`

    where :math:`B` is the biadjacency matrix of the bipartite graph or the adjacency matrix
    of the directed graph.

    * Digraphs
    * Bigraphs

    Parameters
    ----------
    n_components : int (default = 2)
        Dimension of the embedding space.
    regularization : ``None`` or float (default = ``0.01``)
        Add edges of given weight between all pairs of nodes.
    relative_regularization : bool (default = ``True``)
        If ``True``, consider the regularization as relative to the total weight of the graph.
    scaling : float (non-negative, default = ``0.5``)
        Scaling factor :math:`\\alpha` so that each component is divided by
        :math:`(1 - \\lambda)^\\alpha`, with :math:`\\lambda` the corresponding eigenvalue of
        the transition matrix :math:`P`. Require regularization if positive and the graph is not connected.
        The default value :math:`\\alpha=\\frac 1 2` equalizes the energy levels of
        the corresponding mechanical system.
    normalized : bool (default = ``True``)
        If ``True``, normalized the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.
    solver : ``'auto'``, ``'halko'``, ``'lanczos'`` (default = ``'auto'``)
        Which eigenvalue solver to use.

        * ``'auto'`` call the auto_solver function.
        * ``'halko'``: randomized method, fast but less accurate than ``'lanczos'`` for ill-conditioned matrices.
        * ``'lanczos'``: power-iteration based method.

    Attributes
    ----------
    embedding_ : array, shape = (n_row, n_components)
        Embedding of the rows.
    embedding_row_ : array, shape = (n_row, n_components)
        Embedding of the rows (copy of **embedding_**).
    embedding_col_ : array, shape = (n_col, n_components)
        Embedding of the columns.
    eigenvalues_ : array, shape = (n_components)
        Eigenvalues in increasing order (first eigenvalue ignored).
    eigenvectors_ : array, shape = (n, n_components)
        Corresponding eigenvectors.
    regularization_ : ``None`` or float
        Regularization factor added to all pairs of nodes.

    Example
    -------
    >>> from sknetwork.embedding import BiSpectral
    >>> from sknetwork.data import movie_actor
    >>> bispectral = BiSpectral()
    >>> biadjacency = movie_actor()
    >>> embedding = bispectral.fit_transform(biadjacency)
    >>> embedding.shape
    (15, 2)

    References
    ----------
    Belkin, M. & Niyogi, P. (2003). Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
    Neural computation.
    """
    def __init__(self, n_components: int = 2, regularization: Union[None, float] = 0.01,
                 relative_regularization: bool = True, scaling: float = 0.5,
                 normalized: bool = True, solver: str = 'auto'):
        super(BiSpectral, self).__init__(n_components, regularization, relative_regularization, scaling,
                                         normalized, solver)

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiSpectral':
        """Spectral embedding of the bipartite graph considered as undirected, with adjacency matrix:

        :math:`A  = \\begin{bmatrix} 0 & B \\\\ B^T & 0 \\end{bmatrix}`

        where :math:`B` is the biadjacency matrix (or the adjacency matrix of a directed graph).

        Parameters
        ----------
        biadjacency:
            Biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`BiSpectral`
        """
        biadjacency = check_format(biadjacency)
        n_row, _ = biadjacency.shape
        Spectral.fit(self, bipartite2undirected(biadjacency))
        self._split_vars(n_row)

        return self

    def predict(self, adjacency_vectors: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Predict the embedding of new rows, defined by their adjacency vectors.

        Parameters
        ----------
        adjacency_vectors :
            Adjacency vectors of nodes.
            Array of shape (n_col,) (single vector) or (n_vectors, n_col)

        Returns
        -------
        embedding_vectors : np.ndarray
            Embedding of the nodes.
        """
        self._check_fitted()
        n_row, _ = self.embedding_row_.shape
        n_col, _ = self.embedding_col_.shape

        adjacency_vectors = check_adjacency_vector(adjacency_vectors, n_col)
        empty_block = sparse.csr_matrix((adjacency_vectors.shape[0], n_row))
        adjacency_vectors = sparse.bmat([[empty_block, adjacency_vectors]], format='csr')
        embedding_vectors = Spectral.predict(self, adjacency_vectors)

        return embedding_vectors
