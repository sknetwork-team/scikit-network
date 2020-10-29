#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:16:22 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding.base import BaseBiEmbedding
from sknetwork.linalg import SVDSolver, HalkoSVD, LanczosSVD, auto_solver, safe_sparse_dot, diag_pinv, normalize, \
    RegularizedAdjacency, SparseLR
from sknetwork.utils.check import check_format, check_adjacency_vector, check_nonnegative, check_n_components


def set_svd_solver(solver: str, adjacency):
    """SVD solver based on keyword"""
    if solver == 'auto':
        solver: str = auto_solver(adjacency.nnz)
    if solver == 'lanczos':
        solver: SVDSolver = LanczosSVD()
    else:  # pragma: no cover
        solver: SVDSolver = HalkoSVD()
    return solver


class GSVD(BaseBiEmbedding):
    """Graph embedding by Generalized Singular Value Decomposition of the adjacency or biadjacency matrix :math:`A`.
    This is equivalent to the Singular Value Decomposition of the matrix :math:`D_1^{- \\alpha_1}AD_2^{- \\alpha_2}`
    where :math:`D_1, D_2` are the diagonal matrices of row weights and columns weights, respectively, and
    :math:`\\alpha_1, \\alpha_2` are parameters.

    * Graphs
    * Digraphs
    * Bigraphs

    Parameters
    -----------
    n_components : int
        Dimension of the embedding.
    regularization : ``None`` or float (default = ``None``)
        Implicitly add edges of given weight between all pairs of nodes.
    relative_regularization : bool (default = ``True``)
        If ``True``, consider the regularization as relative to the total weight of the graph.
    factor_row : float (default = 0.5)
        Power factor :math:`\\alpha_1` applied to the diagonal matrix of row weights.
    factor_col : float (default = 0.5)
        Power factor :math:`\\alpha_2` applied to the diagonal matrix of column weights.
    factor_singular : float (default = 0.)
        Parameter :math:`\\alpha` applied to the singular values on right singular vectors.
        The embedding of rows and columns are respectively :math:`D_1^{- \\alpha_1}U \\Sigma^{1-\\alpha}` and
        :math:`D_2^{- \\alpha_2}V \\Sigma^\\alpha` where:

        * :math:`U` is the matrix of left singular vectors, shape (n_row, n_components)
        * :math:`V` is the matrix of right singular vectors, shape (n_col, n_components)
        * :math:`\\Sigma` is the diagonal matrix of singular values, shape (n_components, n_components)

    normalized : bool (default = ``True``)
        If ``True``, normalized the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.
    solver : ``'auto'``, ``'halko'``, ``'lanczos'`` or :class:`SVDSolver`
        Which singular value solver to use.

        * ``'auto'``: call the auto_solver function.
        * ``'halko'``: randomized method, fast but less accurate than ``'lanczos'`` for ill-conditioned matrices.
        * ``'lanczos'``: power-iteration based method.
        * :class:`SVDSolver`: custom solver.

    Attributes
    ----------
    embedding_ : np.ndarray, shape = (n1, n_components)
        Embedding of the rows.
    embedding_row_ : np.ndarray, shape = (n1, n_components)
        Embedding of the rows (copy of **embedding_**).
    embedding_col_ : np.ndarray, shape = (n2, n_components)
        Embedding of the columns.
    singular_values_ : np.ndarray, shape = (n_components)
        Singular values.
    singular_vectors_left_ : np.ndarray, shape = (n_row, n_components)
        Left singular vectors.
    singular_vectors_right_ : np.ndarray, shape = (n_col, n_components)
        Right singular vectors.
    regularization_ : ``None`` or float
        Regularization factor added to all pairs of nodes.
    weights_col_ : np.ndarray, shape = (n2)
        Weights applied to columns.

    Example
    -------
    >>> from sknetwork.embedding import GSVD
    >>> from sknetwork.data import karate_club
    >>> gsvd = GSVD()
    >>> adjacency = karate_club()
    >>> embedding = gsvd.fit_transform(adjacency)
    >>> embedding.shape
    (34, 2)

    References
    ----------
    Abdi, H. (2007).
    `Singular value decomposition (SVD) and generalized singular value decomposition.
    <https://www.cs.cornell.edu/cv/ResearchPDF/Generalizing%20The%20Singular%20Value%20Decomposition.pdf>`_
    Encyclopedia of measurement and statistics, 907-912.
    """
    def __init__(self, n_components=2, regularization: Union[None, float] = None, relative_regularization: bool = True,
                 factor_row: float = 0.5, factor_col: float = 0.5, factor_singular: float = 0., normalized: bool = True,
                 solver: Union[str, SVDSolver] = 'auto'):
        super(GSVD, self).__init__()

        self.n_components = n_components
        if regularization == 0:
            self.regularization = None
        else:
            self.regularization = regularization
        self.relative_regularization = relative_regularization
        self.factor_row = factor_row
        self.factor_col = factor_col
        self.factor_singular = factor_singular
        self.normalized = normalized
        self.solver = solver

        self.singular_values_ = None
        self.singular_vectors_left_ = None
        self.singular_vectors_right_ = None
        self.regularization_ = None
        self.weights_col_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'GSVD':
        """Compute the embedding of the graph.

        Parameters
        ----------
        adjacency :
            Adjacency or biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`GSVD`
        """
        adjacency = check_format(adjacency).asfptype()
        n_row, n_col = adjacency.shape
        n_components = check_n_components(self.n_components, min(n_row, n_col) - 1)

        if isinstance(self.solver, str):
            self.solver = set_svd_solver(self.solver, adjacency)
        regularization = self.regularization
        if regularization:
            if self.relative_regularization:
                regularization = regularization * np.sum(adjacency.data) / (n_row * n_col)
            adjacency_reg = RegularizedAdjacency(adjacency, regularization)
        else:
            adjacency_reg = adjacency

        weights_row = adjacency_reg.dot(np.ones(n_col))
        weights_col = adjacency_reg.T.dot(np.ones(n_row))
        diag_row = diag_pinv(np.power(weights_row, self.factor_row))
        diag_col = diag_pinv(np.power(weights_col, self.factor_col))
        self.solver.fit(safe_sparse_dot(diag_row, safe_sparse_dot(adjacency_reg, diag_col)), n_components)

        singular_values = self.solver.singular_values_
        index = np.argsort(-singular_values)
        singular_values = singular_values[index]
        singular_vectors_left = self.solver.singular_vectors_left_[:, index]
        singular_vectors_right = self.solver.singular_vectors_right_[:, index]
        singular_left_diag = sparse.diags(np.power(singular_values, 1 - self.factor_singular))
        singular_right_diag = sparse.diags(np.power(singular_values, self.factor_singular))

        embedding_row = diag_row.dot(singular_vectors_left)
        embedding_col = diag_col.dot(singular_vectors_right)
        embedding_row = singular_left_diag.dot(embedding_row.T).T
        embedding_col = singular_right_diag.dot(embedding_col.T).T

        if self.normalized:
            embedding_row = normalize(embedding_row, p=2)
            embedding_col = normalize(embedding_col, p=2)

        self.embedding_row_ = embedding_row
        self.embedding_col_ = embedding_col
        self.embedding_ = embedding_row
        self.singular_values_ = singular_values
        self.singular_vectors_left_ = singular_vectors_left
        self.singular_vectors_right_ = singular_vectors_right
        self.regularization_ = regularization
        self.weights_col_ = weights_col

        return self

    @staticmethod
    def _check_adj_vector(adjacency_vectors):
        check_nonnegative(adjacency_vectors)

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
        singular_vectors_right = self.singular_vectors_right_
        singular_values = self.singular_values_

        n_row, _ = self.embedding_row_.shape
        n_col, _ = self.embedding_col_.shape

        adjacency_vectors = check_adjacency_vector(adjacency_vectors, n_col)
        self._check_adj_vector(adjacency_vectors)

        # regularization
        if self.regularization_:
            adjacency_vectors = RegularizedAdjacency(adjacency_vectors, self.regularization_)

        # weighting
        weights_row = adjacency_vectors.dot(np.ones(n_col))
        diag_row = diag_pinv(np.power(weights_row, self.factor_row))
        diag_col = diag_pinv(np.power(self.weights_col_, self.factor_col))
        adjacency_vectors = safe_sparse_dot(diag_row, safe_sparse_dot(adjacency_vectors, diag_col))

        # projection in the embedding space
        averaging = adjacency_vectors
        embedding_vectors = diag_row.dot(averaging.dot(singular_vectors_right))

        # scaling
        embedding_vectors /= np.power(singular_values, self.factor_singular)

        if self.normalized:
            embedding_vectors = normalize(embedding_vectors, p=2)

        if embedding_vectors.shape[0] == 1:
            embedding_vectors = embedding_vectors.ravel()

        return embedding_vectors


class SVD(GSVD):
    """Graph embedding by Singular Value Decomposition of the adjacency or biadjacency matrix.

    * Graphs
    * Digraphs
    * Bigraphs

    Parameters
    ----------
    n_components : int
        Dimension of the embedding.
    regularization : ``None`` or float (default = ``None``)
        Implicitly add edges of given weight between all pairs of nodes.
    relative_regularization : bool (default = ``True``)
        If ``True``, consider the regularization as relative to the total weight of the graph.
    factor_singular : float (default = 0.)
        Power factor :math:`\\alpha` applied to the singular values on right singular vectors.
        The embedding of rows and columns are respectively :math:`U \\Sigma^{1-\\alpha}` and
        :math:`V \\Sigma^\\alpha` where:

        * :math:`U` is the matrix of left singular vectors, shape (n_row, n_components)
        * :math:`V` is the matrix of right singular vectors, shape (n_col, n_components)
        * :math:`\\Sigma` is the diagonal matrix of singular values, shape (n_components, n_components)

    normalized : bool (default = ``False``)
        If ``True``, normalized the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.
    solver : ``'auto'``, ``'halko'``, ``'lanczos'`` or :class:`SVDSolver`
        Which singular value solver to use.

        * ``'auto'``: call the auto_solver function.
        * ``'halko'``: randomized method, fast but less accurate than ``'lanczos'`` for ill-conditioned matrices.
        * ``'lanczos'``: power-iteration based method.
        * :class:`SVDSolver`: custom solver.

    Attributes
    ----------
    embedding_ : np.ndarray, shape = (n_row, n_components)
        Embedding of the rows.
    embedding_row_ : np.ndarray, shape = (n_row, n_components)
        Embedding of the rows (copy of **embedding_**).
    embedding_col_ : np.ndarray, shape = (n_col, n_components)
        Embedding of the columns.
    singular_values_ : np.ndarray, shape = (n_components)
        Singular values.
    singular_vectors_left_ : np.ndarray, shape = (n_row, n_components)
        Left singular vectors.
    singular_vectors_right_ : np.ndarray, shape = (n_col, n_components)
        Right singular vectors.
    regularization_ : ``None`` or float
        Regularization factor added to all pairs of nodes.

    Example
    -------
    >>> from sknetwork.embedding import SVD
    >>> from sknetwork.data import karate_club
    >>> svd = SVD()
    >>> adjacency = karate_club()
    >>> embedding = svd.fit_transform(adjacency)
    >>> embedding.shape
    (34, 2)

    References
    ----------
    Abdi, H. (2007).
    `Singular value decomposition (SVD) and generalized singular value decomposition.
    <https://www.cs.cornell.edu/cv/ResearchPDF/Generalizing%20The%20Singular%20Value%20Decomposition.pdf>`_
    Encyclopedia of measurement and statistics.
    """
    def __init__(self, n_components=2, regularization: Union[None, float] = None, relative_regularization: bool = True,
                 factor_singular: float = 0., normalized: bool = False, solver: Union[str, SVDSolver] = 'auto'):
        super(SVD, self).__init__(n_components=n_components, regularization=regularization,
                                  relative_regularization=relative_regularization, factor_singular=factor_singular,
                                  factor_row=0., factor_col=0., normalized=normalized, solver=solver)

    @staticmethod
    def _check_adj_vector(adjacency_vectors: np.ndarray):
        return


class PCA(SVD):
    """Graph embedding by Principal Component Analysis of the adjacency or biadjacency matrix.

    * Graphs
    * Digraphs
    * Bigraphs

    Parameters
    ----------
    n_components : int
        Dimension of the embedding.
    normalized : bool (default = ``False``)
        If ``True``, normalized the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.
    solver : ``'auto'``, ``'halko'``, ``'lanczos'`` or :class:`SVDSolver`
        Which singular value solver to use.

        * ``'auto'``: call the auto_solver function.
        * ``'halko'``: randomized method, fast but less accurate than ``'lanczos'`` for ill-conditioned matrices.
        * ``'lanczos'``: power-iteration based method.
        * :class:`SVDSolver`: custom solver.

    Attributes
    ----------
    embedding_ : np.ndarray, shape = (n_row, n_components)
        Embedding of the rows.
    embedding_row_ : np.ndarray, shape = (n_row, n_components)
        Embedding of the rows (copy of **embedding_**).
    embedding_col_ : np.ndarray, shape = (n_col, n_components)
        Embedding of the columns.
    singular_values_ : np.ndarray, shape = (n_components)
        Singular values.
    singular_vectors_left_ : np.ndarray, shape = (n_row, n_components)
        Left singular vectors.
    singular_vectors_right_ : np.ndarray, shape = (n_col, n_components)
        Right singular vectors.
    regularization_ : ``None`` or float
        Regularization factor added to all pairs of nodes.

    Example
    -------
    >>> from sknetwork.embedding import PCA
    >>> from sknetwork.data import karate_club
    >>> pca = PCA()
    >>> adjacency = karate_club()
    >>> embedding = pca.fit_transform(adjacency)
    >>> embedding.shape
    (34, 2)

    References
    ----------
    Jolliffe, I.T. (2002).
    `Principal Component Analysis`
    Series: Springer Series in Statistics.
    """
    def __init__(self, n_components=2, normalized: bool = False, solver: Union[str, SVDSolver] = 'auto'):
        super(PCA, self).__init__()
        self.n_components = n_components
        self.normalized = normalized
        self.solver = solver

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'PCA':
        """Compute the embedding of the graph.

        Parameters
        ----------
        adjacency :
            Adjacency or biadjacency matrix of the graph.

        Returns
        -------
        self: :class:`PCA`
        """
        adjacency = check_format(adjacency).asfptype()
        n_row, n_col = adjacency.shape
        adjacency_centered = SparseLR(adjacency, (-np.ones(n_row), adjacency.T.dot(np.ones(n_row)) / n_row))

        if isinstance(self.solver, str):
            self.solver = set_svd_solver(self.solver, adjacency)

        svd = self.solver
        svd.fit(adjacency_centered, self.n_components)
        self.embedding_row_ = svd.singular_vectors_left_
        self.embedding_col_ = svd.singular_vectors_right_
        self.embedding_ = svd.singular_vectors_left_
        self.singular_values_ = svd.singular_values_
        self.singular_vectors_left_ = svd.singular_vectors_left_
        self.singular_vectors_right_ = svd.singular_vectors_right_

        return self
