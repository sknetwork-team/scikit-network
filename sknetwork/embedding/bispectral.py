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
from sknetwork.utils.checks import check_format, check_weights


class BiSpectral(BaseEmbedding):
    """
    Graph embedding by Generalized Singular Value Decomposition.

    Solves
    :math:`\\begin{cases} AV = W_1U\\Sigma, \\\\ A^TU = W_2V \\Sigma \\end{cases}`
    where :math:`W_1, W_2` are diagonal matrices of **weights** and **col_weights**.

    The embedding of the rows is :math:`X = U \\phi(\\Sigma)`
    and the embedding of the columns is :math:`Y = V\\psi(\\Sigma)`,
    where :math:`\\phi(\\Sigma)` and :math:`\\psi(\\Sigma)` are diagonal scaling matrices.

    Parameters
    -----------
    embedding_dimension: int
        Dimension of the embedding.
    weights: ``'degree'`` or ``'uniform'`` (default = ``'degree'``)
        Weights of the nodes.

        * ``'degree'``: :math:`W_1 = D`,
        * ``'uniform'``:  :math:`W_1 = I`.

    col_weights: ``None`` or ``'degree'`` or ``'uniform'`` (default= ``None``)
        Weights of the secondary nodes (taken equal to **weights** if ``None``).

        * ``'degree'``: :math:`W_2 = F`,
        * ``'uniform'``:  :math:`W_2 = I`.

    regularization: ``None`` or float (default= ``0.01``)
        Implicitly add edges of given weight between all pairs of nodes.
    relative_regularization : bool (default = ``True``)
        If ``True``, consider the regularization as relative to the total weight of the graph.
    scaling:  ``None`` or ``'multiply'`` or ``'divide'`` or ``'barycenter'`` (default = ``'multiply'``)

        * ``None``: :math:`\\phi(\\Sigma) = \\psi(\\Sigma) = I`,
        * ``'multiply'`` : :math:`\\phi(\\Sigma) = \\psi(\\Sigma) = \\sqrt{\\Sigma}`,
        * ``'divide'``  : :math:`\\phi(\\Sigma)= \\psi(\\Sigma) = (\\sqrt{1 - \\Sigma^2})^{-1}`,
        * ``'barycenter'``  : :math:`\\phi(\\Sigma)= \\Sigma, \\psi(\\Sigma) = I`

    solver: ``'auto'``, ``'halko'``, ``'lanczos'`` or :class:`SVDSolver`
        Which singular value solver to use.

        * ``'auto'``: call the auto_solver function.
        * ``'halko'``: randomized method, fast but less accurate than ``'lanczos'`` for ill-conditioned matrices.
        * ``'lanczos'``: power-iteration based method.
        * :class:`SVDSolver`: custom solver.

    Attributes
    ----------
    row_embedding_ : np.ndarray, shape = (n1, embedding_dimension)
        Embedding of the rows.
    col_embedding_ : np.ndarray, shape = (n2, embedding_dimension)
        Embedding of the columns.
    embedding_ : np.ndarray, shape = (n1 + n2, embedding_dimension)
        Embedding of all nodes (concatenation of the embeddings of rows and columns).
    singular_values_ : np.ndarray, shape = (embedding_dimension)
        Generalized singular values of the adjacency matrix (first singular value ignored).

    Example
    -------
    >>> from sknetwork.data import simple_bipartite_graph
    >>> biadjacency = simple_bipartite_graph()
    >>> bispectral = BiSpectral()
    >>> embedding = bispectral.fit_transform(biadjacency)
    >>> embedding.shape
    (9, 2)

    References
    ----------
    Abdi, H. (2007).
    `Singular value decomposition (SVD) and generalized singular value decomposition.
    <https://www.cs.cornell.edu/cv/ResearchPDF/Generalizing%20The%20Singular%20Value%20Decomposition.pdf>`_
    Encyclopedia of measurement and statistics, 907-912.
    """

    def __init__(self, embedding_dimension=2, weights='degree', col_weights=None,
                 regularization: Union[None, float] = 0.01, relative_regularization: bool = True,
                 scaling: Union[None, str] = 'multiply', solver: Union[str, SVDSolver] = 'auto'):
        super(BiSpectral, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.weights = weights
        if col_weights is None:
            col_weights = weights
        self.col_weights = col_weights
        self.regularization = regularization
        self.relative_regularization = relative_regularization
        self.scaling = scaling

        if scaling == 'divide':
            if weights != 'degree' or col_weights != 'degree':
                self.scaling = None
                warnings.warn(Warning("The scaling 'divide' is valid only with ``weights = 'degree'`` and "
                                      "``col_weights = 'degree'``. It will be ignored."))

        if solver == 'halko':
            self.solver: SVDSolver = HalkoSVD()
        elif solver == 'lanczos':
            self.solver: SVDSolver = LanczosSVD()
        else:
            self.solver = solver

        self.row_embedding_ = None
        self.col_embedding_ = None
        self.embedding_ = None
        self.singular_values_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'BiSpectral':
        """
        Computes the generalized SVD of the adjacency matrix.

        Parameters
        ----------
        adjacency: array-like, shape = (n1, n2)
            Adjacency matrix, where n1 = n2 is the number of nodes for a standard graph,
            n1, n2 are the number of nodes in each part for a bipartite graph.

        Returns
        -------
        self: :class:`BiSpectral`
        """
        adjacency = check_format(adjacency).asfptype()
        n1, n2 = adjacency.shape

        if self.solver == 'auto':
            solver = auto_solver(adjacency.nnz)
            if solver == 'lanczos':
                self.solver: SVDSolver = LanczosSVD()
            else:
                self.solver: SVDSolver = HalkoSVD()

        total_weight = adjacency.dot(np.ones(n2)).sum()
        regularization = self.regularization
        if regularization:
            if self.relative_regularization:
                regularization = regularization * total_weight / (n1 * n2)
            adjacency = SparseLR(adjacency, [(regularization * np.ones(n1), np.ones(n2))])

        w_row = check_weights(self.weights, adjacency)
        w_col = check_weights(self.col_weights, adjacency.T)
        diag_row = diag_pinv(np.sqrt(w_row))
        diag_col = diag_pinv(np.sqrt(w_col))

        normalized_adj = safe_sparse_dot(diag_row, safe_sparse_dot(adjacency, diag_col))

        # svd
        if self.embedding_dimension >= min(n1, n2) - 1:
            n_components = min(n1, n2) - 1
            warnings.warn(Warning("The dimension of the embedding must be less than the number of rows "
                                  "and the number of columns. Changed accordingly."))
        else:
            n_components = self.embedding_dimension + 1
        self.solver.fit(normalized_adj, n_components)

        index = np.argsort(-self.solver.singular_values_)
        self.singular_values_ = self.solver.singular_values_[index[1:]]
        self.row_embedding_ = diag_row.dot(self.solver.left_singular_vectors_[:, index[1:]])
        self.col_embedding_ = diag_col.dot(self.solver.right_singular_vectors_[:, index[1:]])

        if self.scaling:
            if self.scaling == 'multiply':
                self.row_embedding_ *= np.sqrt(self.singular_values_)
                self.col_embedding_ *= np.sqrt(self.singular_values_)
            elif self.scaling == 'divide':
                energy_levels: np.ndarray = np.sqrt(1 - np.clip(self.singular_values_, 0, 1) ** 2)
                energy_levels[energy_levels > 0] = 1 / energy_levels[energy_levels > 0]
                self.row_embedding_ *= energy_levels
                self.col_embedding_ *= energy_levels
            elif self.scaling == 'barycenter':
                self.row_embedding_ *= self.singular_values_
            else:
                warnings.warn(Warning("The scaling must be 'multiply' or 'divide' or 'barycenter'. No scaling done."))

        self.embedding_ = np.vstack((self.row_embedding_, self.col_embedding_))
        return self
