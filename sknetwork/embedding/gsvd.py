#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:16:22 2018
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import sparse, linalg
from sknetwork.utils import SparseLR
from sknetwork.utils.algorithm_base_class import Algorithm
from sknetwork.utils.checks import check_format, check_weights, check_random_state
from sknetwork.utils.randomized_matrix_factorization import randomized_svd, safe_sparse_dot
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
    feature_weights: ``'degree'`` or ``'uniform'``
        Default weighting for the columns.
    low_rank_regularization: ``None`` or float (default=0.01)
        Implicitly add edges of given weight between all pairs of nodes.
    energy_scaling: bool (default=True)
        If ``True``, rescales each column of the embedding by dividing it by :math:`\\sqrt{1-\\sigma_i^2}`.
        Only valid if ``weights == 'degree'`` and ``feature_weights == 'degree'``.

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
    >>> from sknetwork.toy_graphs import movie_actor_graph
    >>> graph = movie_actor_graph()
    >>> gsvd = GSVD(embedding_dimension=2)
    >>> gsvd.fit(graph)
    GSVD(embedding_dimension=2, weights='degree', feature_weights='degree', low_rank_regularization=0.01,\
 energy_scaling=True)
    >>> gsvd.embedding_.shape
    (15, 2)

    References
    ----------
    * Abdi, H. (2007). Singular value decomposition (SVD) and generalized singular value decomposition.
      Encyclopedia of measurement and statistics, 907-912.
      https://www.cs.cornell.edu/cv/ResearchPDF/Generalizing%20The%20Singular%20Value%20Decomposition.pdf
    """

    def __init__(self, embedding_dimension=2, weights='degree', feature_weights='degree',
                 low_rank_regularization: Union[None, float] = 0.01, energy_scaling: bool = True):
        self.embedding_dimension = embedding_dimension
        self.weights = weights
        self.feature_weights = feature_weights
        self.low_rank_regularization = low_rank_regularization
        self.energy_scaling = energy_scaling
        self.embedding_ = None
        self.features_ = None
        self.singular_values_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], weights=None, feature_weights=None,
            randomized_decomposition: bool = True, n_iter='auto', power_iteration_normalizer: Union[str, None] = 'auto',
            random_state=None) -> 'GSVD':
        """Fits the model from data in adjacency_matrix.

        Parameters
        ----------
        adjacency: array-like, shape = (n, m)
            Adjacency matrix, where n = m is the number of nodes for a standard directed or undirected graph,
            n is the cardinal of V1 and m is the cardinal of V2 for a bipartite graph.
        weights: ``None``, ``'degree'``, ``'uniform'`` or customized numpy array
            If None, it will use the default value.
        feature_weights: ``None``, ``'degree'``, ``'uniform'`` or customized numpy array
            If None, it will use the default value.
        randomized_decomposition:
            whether to use a randomized (and faster) svd method or the standard scipy one.
        n_iter: int or ``'auto'`` (default is ``'auto'``)
            See :meth:`sknetwork.embedding.randomized_range_finder`
        power_iteration_normalizer: ``'auto'`` (default), ``'QR'``, ``'LU'``, ``None``
            See :meth:`sknetwork.embedding.randomized_range_finder`
        random_state: int, RandomState instance or ``None``, optional (default= ``None``)
            See :meth:`sknetwork.embedding.randomized_range_finder`

        Returns
        -------
        self: :class:`GSVD`
        """
        random_state = check_random_state(random_state)
        adjacency = check_format(adjacency)
        n_nodes, m_nodes = adjacency.shape
        if self.low_rank_regularization:
            adjacency = SparseLR(adjacency, [(self.low_rank_regularization * np.ones(n_nodes), np.ones(m_nodes))])
        total_weight = adjacency.dot(np.ones(m_nodes)).sum()

        if weights is None:
            weights = self.weights
        w_samp = check_weights(weights, adjacency)

        if feature_weights is None:
            feature_weights = self.feature_weights
        w_feat = check_weights(feature_weights, adjacency.T)

        # pseudo inverse square-root out-degree matrix
        diag_samp = sparse.diags(np.sqrt(w_samp), shape=(n_nodes, n_nodes), format='csr')
        diag_samp.data = 1 / diag_samp.data
        # pseudo inverse square-root in-degree matrix
        diag_feat = sparse.diags(np.sqrt(w_feat), shape=(m_nodes, m_nodes), format='csr')
        diag_feat.data = 1 / diag_feat.data

        normalized_adj = safe_sparse_dot(diag_samp, safe_sparse_dot(adjacency, diag_feat))

        if randomized_decomposition:
            u, sigma, vt = randomized_svd(normalized_adj, self.embedding_dimension,
                                          n_iter=n_iter,
                                          power_iteration_normalizer=power_iteration_normalizer,
                                          random_state=random_state)
        else:
            u, sigma, vt = linalg.svds(normalized_adj, self.embedding_dimension)

        self.singular_values_ = sigma
        self.embedding_ = np.sqrt(total_weight) * diag_samp.dot(u) * sigma
        self.features_ = np.sqrt(total_weight) * diag_feat.dot(vt.T)

        if self.energy_scaling and weights == 'degree' and feature_weights == 'degree':
            energy_levels: np.ndarray = np.sqrt(1 - np.clip(sigma, 0, 1) ** 2)
            energy_levels[energy_levels > 0] = 1 / energy_levels[energy_levels > 0]
            self.embedding_ *= energy_levels
            self.features_ *= energy_levels

        return self
