#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:16:22 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

import warnings
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding.base import BaseEmbedding
from sknetwork.linalg import SparseLR, SVDSolver, HalkoSVD, LanczosSVD, auto_solver, safe_sparse_dot, diag_pinv
from sknetwork.utils.check import check_format


class GSVD(BaseEmbedding):
    """Graph embedding by Generalized Singular Value Decomposition of the adjacency or biadjacency matrix :math:`A`.
    This is equivalent to the Singular Value Decomposition of the matrix :math:`D_1^{- \\alpha_1}AD_2^{- \\alpha_2}`
    where :math:`D_1, D_2` are the diagonal matrices of row weights and columns weights, respectively, and
    :math:`\\alpha_1, \\alpha_2` are parameters.

    Parameters
    -----------
    n_components: int
        Dimension of the embedding.
    regularization: ``None`` or float (default = ``None``)
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

    normalize : bool (default = ``True``)
        If ``True``, normalize the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.
    solver: ``'auto'``, ``'halko'``, ``'lanczos'`` or :class:`SVDSolver`
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
    >>> gsvd = GSVD(3)
    >>> embedding = gsvd.fit_transform(np.ones((6,4)))
    >>> embedding.shape
    (6, 3)

    References
    ----------
    Abdi, H. (2007).
    `Singular value decomposition (SVD) and generalized singular value decomposition.
    <https://www.cs.cornell.edu/cv/ResearchPDF/Generalizing%20The%20Singular%20Value%20Decomposition.pdf>`_
    Encyclopedia of measurement and statistics, 907-912.
    """

    def __init__(self, n_components=2, regularization: Union[None, float] = None, relative_regularization: bool = True,
                 factor_row: float = 0.5, factor_col: float = 0.5, factor_singular: float = 0., normalize: bool = True,
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
        self.normalize = normalize
        if solver == 'halko':
            self.solver: SVDSolver = HalkoSVD()
        elif solver == 'lanczos':
            self.solver: SVDSolver = LanczosSVD()
        else:
            self.solver = solver

        self.embedding_ = None
        self.embedding_row_ = None
        self.embedding_col_ = None
        self.singular_values_ = None
        self.singular_vectors_left_ = None
        self.singular_vectors_right_ = None
        self.regularization_ = None
        self.weights_col_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'GSVD':
        """
        Compute the GSVD of the adjacency or biadjacency matrix.

        Parameters
        ----------
        adjacency: array-like, shape = (n1, n2)
            Adjacency matrix, where n1 = n2 is the number of nodes for a standard graph,
            n1, n2 are the number of nodes in each part for a bipartite graph.

        Returns
        -------
        self: :class:`GSVD`
        """
        adjacency = check_format(adjacency).asfptype()
        n1, n2 = adjacency.shape

        if self.n_components >= min(n1, n2) - 1:
            n_components = min(n1, n2) - 1
            warnings.warn(Warning("The dimension of the embedding must be less than the number of rows "
                                  "and the number of columns. Changed accordingly."))
        else:
            n_components = self.n_components

        if self.solver == 'auto':
            solver = auto_solver(adjacency.nnz)
            if solver == 'lanczos':
                self.solver: SVDSolver = LanczosSVD()
            else:
                self.solver: SVDSolver = HalkoSVD()

        regularization = self.regularization
        if regularization:
            if self.relative_regularization:
                regularization = regularization * np.sum(adjacency.data) / (n1 * n2)
            adjacency_reg = SparseLR(adjacency, [(regularization * np.ones(n1), np.ones(n2))])
        else:
            adjacency_reg = adjacency

        weights_row = adjacency_reg.dot(np.ones(n2))
        weights_col = adjacency_reg.T.dot(np.ones(n1))
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

        if self.normalize:
            embedding_row = diag_pinv(np.linalg.norm(embedding_row, axis=1)).dot(embedding_row)
            embedding_col = diag_pinv(np.linalg.norm(embedding_col, axis=1)).dot(embedding_col)

        self.embedding_row_ = embedding_row
        self.embedding_col_ = embedding_col
        self.embedding_ = embedding_row
        self.singular_values_ = singular_values
        self.singular_vectors_left_ = singular_vectors_left
        self.singular_vectors_right_ = singular_vectors_right
        self.regularization_ = regularization
        self.weights_col_ = weights_col

        return self

    def predict(self, adjacency_vectors: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Predict the embedding of new rows, defined by their adjacency vectors.

        Parameters
        ----------
        adjacency_vectors : array, shape (n_col,) (single vector) or (n_vectors, n_col)
            Adjacency vectors of nodes.

        Returns
        -------
        embedding_vectors : array, shape (n_components,) (single vector) or (n_vectors, n_components)
            Embedding of the nodes.
        """
        singular_vectors_right = self.singular_vectors_right_
        if singular_vectors_right is None:
            raise ValueError("This instance of SVD embedding is not fitted yet."
                             " Call 'fit' with appropriate arguments before using this method.")
        singular_values = self.singular_values_

        n1, _ = self.embedding_row_.shape
        n2, _ = self.embedding_col_.shape

        if isinstance(adjacency_vectors, sparse.csr_matrix):
            adjacency_vectors = np.array(adjacency_vectors.todense())

        single_vector = False
        if len(adjacency_vectors.shape) == 1:
            single_vector = True
            adjacency_vectors = adjacency_vectors.reshape(1, -1)

        if adjacency_vectors.shape[1] != n2:
            raise ValueError('The adjacency vector must be of length equal to the number of columns of the '
                             'biadjacency matrix.')
        elif not np.all(adjacency_vectors >= 0):
            raise ValueError('The adjacency vector must be non-negative.')

        # regularization
        adjacency_vector_reg = adjacency_vectors.astype(float)
        if self.regularization_:
            adjacency_vector_reg += self.regularization_

        # weighting
        weights_row = adjacency_vector_reg.dot(np.ones(n2))
        diag_row = diag_pinv(np.power(weights_row, self.factor_row))
        diag_col = diag_pinv(np.power(self.weights_col_, self.factor_col))
        adjacency_vector_reg = diag_row.dot(diag_col.dot(adjacency_vector_reg.T).T)

        # projection in the embedding space
        diag = diag_pinv(np.sum(adjacency_vector_reg, axis=1))
        averaging = diag.dot(adjacency_vector_reg)
        embedding_vectors = diag_row.dot(averaging.dot(singular_vectors_right))

        # scaling
        embedding_vectors /= np.power(singular_values, self.factor_singular)

        if self.normalize:
            embedding_vectors = diag_pinv(np.linalg.norm(embedding_vectors, axis=1)).dot(embedding_vectors)

        if single_vector:
            embedding_vectors = embedding_vectors.ravel()

        return embedding_vectors


class SVD(GSVD):
    """Graph embedding by Singular Value Decomposition of the adjacency or biadjacency matrix.

    Parameters
    -----------
    n_components: int
        Dimension of the embedding.
    regularization: ``None`` or float (default = ``None``)
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

    normalize : bool (default = ``False``)
        If ``True``, normalize the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.
    solver: ``'auto'``, ``'halko'``, ``'lanczos'`` or :class:`SVDSolver`
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

    Example
    -------
    >>> svd = SVD(3)
    >>> embedding = svd.fit_transform(np.ones((6,5)))
    >>> embedding.shape
    (6, 3)

    References
    ----------
    Abdi, H. (2007).
    `Singular value decomposition (SVD) and generalized singular value decomposition.
    <https://www.cs.cornell.edu/cv/ResearchPDF/Generalizing%20The%20Singular%20Value%20Decomposition.pdf>`_
    Encyclopedia of measurement and statistics, 907-912.
    """

    def __init__(self, n_components=2, regularization: Union[None, float] = None, relative_regularization: bool = True,
                 factor_singular: float = 0., normalize: bool = False, solver: Union[str, SVDSolver] = 'auto'):
        GSVD.__init__(self, n_components, regularization, relative_regularization, factor_singular, 0, 0, normalize,
                      solver)

        self.embedding_ = None
        self.embedding_row_ = None
        self.embedding_col_ = None
        self.singular_values_ = None
        self.singular_vectors_left_ = None
        self.singular_vectors_right_ = None
        self.regularization_ = None
