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
from sknetwork.utils.checks import check_format


class SVD(BaseEmbedding):
    """Graph embedding by Singular Value Decomposition of the adjacency or biadjacency matrix.

    Parameters
    -----------
    n_components: int
        Dimension of the embedding.
    regularization: ``None`` or float (default = ``None``)
        Implicitly add edges of given weight between all pairs of nodes.
    relative_regularization : bool (default = ``True``)
        If ``True``, consider the regularization as relative to the total weight of the graph.
    singular_right : float (default = 0.)
        Parameter :math:`\\alpha` applied to the singular values on right singular vectors.
        The embedding of rows and columns are respectively :math:`U \\Sigma^{1-\\alpha}` and
        :math:`V \\Sigma^\\alpha` where:

        * :math:`U` is the matrix of left singular vectors, shape (n_row, n_components)
        * :math:`V` is the matrix of right singular vectors, shape (n_col, n_components)
        * :math:`\\Sigma` is the diagonal matrix of singular values, shape (n_components, n_components)

    normalize : bool (default = ``True``)
        If ``True``, normalizes the embedding so that each vector has norm 1 in the embedding space, i.e.,
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
        Embedding of the rows (copy of 'embedding_').
    embedding_col_ : np.ndarray, shape = (n2, n_components)
        Embedding of the columns.
    singular_values_ : np.ndarray, shape = (n_components)
        Singular values.
    singular_vectors_left_ : np.ndarray, shape = (n_row, n_components)
        Left singular vectors.
    singular_vectors_right_ : np.ndarray, shape = (n_col, n_components)
        Right singular vectors.

    Example
    -------
    >>> svd = SVD()
    >>> embedding = svd.fit_transform(np.ones((5,4)))
    >>> embedding.shape
    (5, 2)

    References
    ----------
    Abdi, H. (2007).
    `Singular value decomposition (SVD) and generalized singular value decomposition.
    <https://www.cs.cornell.edu/cv/ResearchPDF/Generalizing%20The%20Singular%20Value%20Decomposition.pdf>`_
    Encyclopedia of measurement and statistics, 907-912.
    """

    def __init__(self, n_components=2, regularization: Union[None, float] = None, relative_regularization: bool = True,
                 singular_right: float = 0., normalize: bool = True, solver: Union[str, SVDSolver] = 'auto'):
        super(SVD, self).__init__()

        self.n_components = n_components
        if regularization == 0:
            self.regularization = None
        else:
            self.regularization = regularization
        self.relative_regularization = relative_regularization
        self.singular_right = singular_right
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

    @staticmethod
    def weight_matrices(adjacency):
        """

        Parameters
        ----------
        adjacency

        Returns
        -------

        """
        n1, n2 = adjacency.shape
        return sparse.eye(n1, format='csr'), sparse.eye(n2, format='csr')

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'SVD':
        """
        Computes the SVD of the adjacency or biadjacency matrix.

        Parameters
        ----------
        adjacency: array-like, shape = (n1, n2)
            Adjacency matrix, where n1 = n2 is the number of nodes for a standard graph,
            n1, n2 are the number of nodes in each part for a bipartite graph.

        Returns
        -------
        self: :class:`SVD`
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

        diag_row, diag_col = self.weight_matrices(adjacency)

        regularization = self.regularization
        if regularization:
            if self.relative_regularization:
                regularization = regularization * np.sum(adjacency.data) / (n1 * n2)
            adjacency_reg = SparseLR(adjacency, [(regularization * np.ones(n1), np.ones(n2))])
            self.solver.fit(safe_sparse_dot(diag_row, safe_sparse_dot(adjacency_reg, diag_col)), n_components)
        else:
            self.solver.fit(safe_sparse_dot(diag_row, safe_sparse_dot(adjacency, diag_col)), n_components)

        singular_values = self.solver.singular_values_
        index = np.argsort(-singular_values)
        singular_values = singular_values[index]
        singular_vectors_left = self.solver.singular_vectors_left_[:, index]
        singular_vectors_right = self.solver.singular_vectors_right_[:, index]
        singular_left_diag = sparse.diags(np.power(singular_values, 1 - self.singular_right))
        singular_right_diag = sparse.diags(np.power(singular_values, self.singular_right))

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

        return self


class GSVD(SVD):
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
        Parameter :math:`\\alpha_1` applied to the diagonal matrix of row weights.
    factor_col : float (default = 0.5)
        Parameter :math:`\\alpha_2` applied to the diagonal matrix of column weights.
    singular_right : float (default = 0.)
        Parameter :math:`\\alpha` applied to the singular values on right singular vectors.
        The embedding of rows and columns are respectively :math:`D_1^{- \\alpha_1}U \\Sigma^{1-\\alpha}` and
        :math:`D_2^{- \\alpha_1}V \\Sigma^\\alpha` where:

        * :math:`U` is the matrix of left singular vectors, shape (n_row, n_components)
        * :math:`V` is the matrix of right singular vectors, shape (n_col, n_components)
        * :math:`\\Sigma` is the diagonal matrix of singular values, shape (n_components, n_components)

    normalize : bool (default = ``True``)
        If ``True``, normalizes the embedding so that each vector has norm 1 in the embedding space, i.e.,
        each vector lies on the unit sphere.
    solver: ``'auto'``, ``'halko'``, ``'lanczos'`` or :class:`SVDSolver`
        Which singular value solver to use.

        * ``'auto'``: call the auto_solver function.
        * ``'halko'``: randomized method, fast but less accurate than ``'lanczos'`` for ill-conditioned matrices.
        * ``'lanczos'``: power-iteration based method.
        * :class:`SVDSolver`: custom solver.

    Example
    -------
    >>> gsvd = GSVD()
    >>> embedding = gsvd.fit_transform(np.ones((5,4)))
    >>> embedding.shape
    (5, 2)

    References
    ----------
    Abdi, H. (2007).
    `Singular value decomposition (SVD) and generalized singular value decomposition.
    <https://www.cs.cornell.edu/cv/ResearchPDF/Generalizing%20The%20Singular%20Value%20Decomposition.pdf>`_
    Encyclopedia of measurement and statistics, 907-912.
    """

    def __init__(self, n_components=2, regularization: Union[None, float] = None, relative_regularization: bool = True,
                 factor_row: float = 0.5, factor_col: float = 0.5, singular_right: float = 0., normalize: bool = True,
                 solver: Union[str, SVDSolver] = 'auto'):
        super(GSVD, self).__init__(n_components, regularization, relative_regularization, singular_right, normalize,
                                   solver)
        self.factor_row = factor_row
        self.factor_col = factor_col

    def weight_matrices(self, adjacency):
        """

        Parameters
        ----------
        adjacency

        Returns
        -------

        """
        n1, n2 = adjacency.shape
        weights_row = adjacency.dot(np.ones(n2))
        weights_col = adjacency.T.dot(np.ones(n1))
        diag_row = diag_pinv(np.power(weights_row, self.factor_row))
        diag_col = diag_pinv(np.power(weights_col, self.factor_col))
        return diag_row, diag_col
