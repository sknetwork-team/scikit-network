#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:16:22 2018
@author: Nathan de Lara <ndelara@enst.fr>
@author: Thomas Bonald <bonald@enst.fr>
"""

from typing import Union
import warnings

import numpy as np
from scipy import sparse

from sknetwork.linalg import SparseLR, SVDSolver, HalkoSVD, LanczosSVD, auto_solver, safe_sparse_dot
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format, check_weights


class SVD(Algorithm):
    """
    Graph embedding by Generalized Singular Value Decomposition.

    Setting **weights** and **feature_weights** to ``'uniform'`` leads to the standard SVD.

    Parameters
    -----------
    embedding_dimension: int
        Dimension of the embedding.
    weights: ``'degree'`` or ``'uniform'`` (default= ``'degree'``)
        Weights of the nodes.
    feature_weights: ``'degree'`` or ``'uniform'`` (default= ``'degree'``)
        Weights of the feature nodes.
    regularization: ``None`` or float (default= ``0.01``)
        Implicitly add edges of given weight between all pairs of nodes.
    energy_scaling: bool (default= ``True``)
        If ``True``, rescales each dimension of the embedding by the corresponding energy.
        Only valid if **weights** = ``'degree'`` and **feature_weights** = ``'degree'``.

    Attributes
    ----------
    embedding_ : np.ndarray, shape = (n_samples, embedding_dimension)
        Embedding of the nodes (rows of the adjacency matrix)
    coembedding_ : np.ndarray, shape = (n_features, embedding_dimension)
        Embedding of the feature nodes (columns of the adjacency matrix)
    singular_values_ : np.ndarray, shape = (embedding_dimension)
        Generalized singular values of the adjacency matrix

    Example
    -------
    >>> from sknetwork.toy_graphs import movie_actor
    >>> adjacency: sparse.csr_matrix = movie_actor()
    >>> svd = SVD(embedding_dimension=2)
    >>> embedding = svd.fit(adjacency).embedding_
    >>> embedding.shape
    (15, 2)

    References
    ----------
    Abdi, H. (2007). Singular value decomposition (SVD) and generalized singular value decomposition.
    Encyclopedia of measurement and statistics, 907-912.
    https://www.cs.cornell.edu/cv/ResearchPDF/Generalizing%20The%20Singular%20Value%20Decomposition.pdf
    """

    def __init__(self, embedding_dimension=2, weights='degree', feature_weights='degree',
                 regularization: Union[None, float] = 0.01, energy_scaling: bool = True,
                 solver: Union[str, SVDSolver] = 'auto'):
        self.embedding_dimension = embedding_dimension
        self.weights = weights
        self.feature_weights = feature_weights
        self.regularization = regularization
        self.energy_scaling = energy_scaling

        if energy_scaling:
            if weights != 'degree' or feature_weights != 'degree':
                warnings.warn(Warning("The option energy_scaling is valid only with ``weights = 'degree'`` and "
                                      "``feature_weights = 'degree'``. It will be ignored."))

        if solver == 'halko':
            self.solver: SVDSolver = HalkoSVD()
        elif solver == 'lanczos':
            self.solver: SVDSolver = LanczosSVD()
        else:
            self.solver = solver

        self.embedding_ = None
        self.coembedding_ = None
        self.singular_values_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'SVD':
        """
        Computes the generalized SVD of the adjacency matrix.

        Parameters
        ----------
        adjacency: array-like, shape = (n, p)
            Adjacency matrix, where n = p is the number of nodes for a standard directed or undirected graph,
            n, p are the number of nodes in each part for a bipartite graph.

        Returns
        -------
        self: :class:`SVD`
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
        w_feat = check_weights(self.feature_weights, adjacency.T)

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

        index = np.argsort(-self.solver.singular_values_)
        self.singular_values_ = self.solver.singular_values_[index[1:]]
        self.embedding_ = np.sqrt(total_weight) * diag_samp.dot(self.solver.left_singular_vectors_[:, index[1:]])
        self.coembedding_ = np.sqrt(total_weight) * diag_feat.dot(self.solver.right_singular_vectors_[:, index[1:]])

        # rescale to get barycenter property
        self.embedding_ *= self.singular_values_

        if self.energy_scaling and self.weights == 'degree' and self.feature_weights == 'degree':
            energy_levels: np.ndarray = np.sqrt(1 - np.clip(self.singular_values_, 0, 1) ** 2)
            energy_levels[energy_levels > 0] = 1 / energy_levels[energy_levels > 0]
            self.embedding_ *= energy_levels
            self.coembedding_ *= energy_levels

        return self
