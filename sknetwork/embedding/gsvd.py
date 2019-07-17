#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:16:22 2018
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse
from sknetwork.linalg import SparseLR
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format, check_weights
from sknetwork.linalg import SVDSolver, HalkoSVD, LanczosSVD, auto_solver, safe_sparse_dot
from typing import Union


class GSVD(Algorithm):
    """Generalized Singular Value Decomposition for non-linear dimensionality reduction.

    Setting ``weights`` and ``feature_weights`` to ``'uniform'`` leads to the standard SVD.

    Parameters
    -----------
    embedding_dimension: int
        The dimension of the projected subspace.
    weights: ``'degree'`` or ``'uniform'``
        Default weighting for the rows.
    regularization: ``None`` or float (default=0.01)
        Implicitly add edges of given weight between all pairs of nodes.
    energy_scaling: bool (default=True)
        If ``True``, rescales each column of the embedding by dividing it by :math:`\\sqrt{1-\\sigma_i^2}`.
        Only valid if ``weights == 'degree'``.

    Attributes
    ----------
    embedding_ : np.ndarray, shape = (n_samples, embedding_dimension)
        Embedding of the samples (rows of the training matrix)
    features_ : np.ndarray, shape = (n_features, embedding_dimension)
        Embedding of the features (columns of the training matrix)
    singular_values_ : np.ndarray, shape = (embedding_dimension)
        Singular values of the training matrix

    Example
    -------
    >>> from sknetwork.toy_graphs import movie_actor
    >>> adjacency = movie_actor()
    >>> gsvd = GSVD(embedding_dimension=2)
    >>> gsvd.fit(adjacency)
    GSVD(embedding_dimension=2, weights='degree', regularization=0.01, energy_scaling=True, solver=LanczosSVD())
    >>> gsvd.embedding_.shape
    (15, 2)

    References
    ----------
    * Abdi, H. (2007). Singular value decomposition (SVD) and generalized singular value decomposition.
      Encyclopedia of measurement and statistics, 907-912.
      https://www.cs.cornell.edu/cv/ResearchPDF/Generalizing%20The%20Singular%20Value%20Decomposition.pdf
    """

    def __init__(self, embedding_dimension=2, weights='degree', regularization: Union[None, float] = 0.01,
                 energy_scaling: bool = True, solver: Union[str, SVDSolver] = 'auto'):
        self.embedding_dimension = embedding_dimension
        self.weights = weights
        self.regularization = regularization
        self.energy_scaling = energy_scaling
        if solver == 'halko':
            self.solver: SVDSolver = HalkoSVD()
        elif solver == 'lanczos':
            self.solver: SVDSolver = LanczosSVD()
        else:
            self.solver = solver

        self.embedding_ = None
        self.features_ = None
        self.singular_values_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'GSVD':
        """Fits the model from data in adjacency_matrix.

        Parameters
        ----------
        adjacency: array-like, shape = (n, p)
            Adjacency matrix, where n = p is the number of nodes for a standard directed or undirected adjacency,
            n, p are the number of nodes in each part for a biadjacency adjacency.

        Returns
        -------
        self: :class:`GSVD`
        """
        adjacency = check_format(adjacency)
        n, p = adjacency.shape

        if self.solver == 'auto':
            solver = auto_solver(adjacency.nnz)
            if solver == 'lanczos':
                self.solver: SVDSolver = LanczosSVD()
            else:
                self.solver: SVDSolver = HalkoSVD()

        if self.regularization:
            adjacency = SparseLR(adjacency, [(self.regularization * np.ones(n), np.ones(p))])
        total_weight = adjacency.dot(np.ones(p)).sum()

        w_samp = check_weights(self.weights, adjacency)
        w_feat = check_weights(self.weights, adjacency.T)

        # pseudo inverse square-root out-degree matrix
        diag_samp = sparse.diags(np.sqrt(w_samp), shape=(n, n), format='csr')
        diag_samp.data = 1 / diag_samp.data
        # pseudo inverse square-root in-degree matrix
        diag_feat = sparse.diags(np.sqrt(w_feat), shape=(p, p), format='csr')
        diag_feat.data = 1 / diag_feat.data

        normalized_adj = safe_sparse_dot(diag_samp, safe_sparse_dot(adjacency, diag_feat))

        # svd
        n_components = min(self.embedding_dimension + 1, min(n, p) - 1)
        self.solver.fit(normalized_adj, n_components)

        self.singular_values_ = self.solver.singular_values_[1:]
        self.embedding_ = np.sqrt(total_weight) * diag_samp.dot(self.solver.left_singular_vectors_[:, 1:])
        self.features_ = np.sqrt(total_weight) * diag_feat.dot(self.solver.right_singular_vectors_[:, 1:])

        # rescale to get barycenter property
        self.embedding_ *= self.singular_values_

        if self.energy_scaling and self.weights == 'degree':
            energy_levels: np.ndarray = np.sqrt(1 - np.clip(self.singular_values_, 0, 1) ** 2)
            energy_levels[energy_levels > 0] = 1 / energy_levels[energy_levels > 0]
            self.embedding_ *= energy_levels
            self.features_ *= energy_levels

        return self
