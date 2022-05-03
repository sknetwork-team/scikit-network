#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC

import numpy as np
from scipy import sparse

from sknetwork.linalg.normalization import normalize
from sknetwork.utils.base import Algorithm
from sknetwork.utils.membership import membership_matrix


class BaseClustering(Algorithm, ABC):
    """Base class for clustering algorithms.

    Attributes
    ----------
    labels_ : np.ndarray
        Labels of the nodes (rows for bipartite graphs)
    labels_row_ : np.ndarray
        Labels of the rows (for bipartite graphs).
    labels_col_ : np.ndarray
        Labels of the columns (for bipartite graphs, in case of co-clustering).
    membership_ : sparse.csr_matrix
        Membership matrix of the nodes, shape (n_nodes, n_clusters).
    membership_row_ : sparse.csr_matrix
        Membership matrix of the rows (for bipartite graphs).
    membership_col_ : sparse.csr_matrix
        Membership matrix of the columns (for bipartite graphs, in case of co-clustering).
    aggregate_ : sparse.csr_matrix
        Aggregate adjacency matrix or biadjacency matrix between clusters.
    """
    def __init__(self, sort_clusters: bool = True, return_membership: bool = False, return_aggregate: bool = False):
        self.sort_clusters = sort_clusters
        self.return_membership = return_membership
        self.return_aggregate = return_aggregate
        self._init_vars()

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
        self.fit(*args, **kwargs)
        return self.labels_

    def _init_vars(self):
        """Init variables."""
        self.labels_ = None
        self.labels_row_ = None
        self.labels_col_ = None
        self.membership_ = None
        self.membership_row_ = None
        self.membership_col_ = None
        self.aggregate_ = None
        self.bipartite = None
        return self

    def _split_vars(self, shape):
        """Split labels_ into labels_row_ and labels_col_"""
        n_row = shape[0]
        self.labels_row_ = self.labels_[:n_row]
        self.labels_col_ = self.labels_[n_row:]
        self.labels_ = self.labels_row_
        return self

    def _secondary_outputs(self, input_matrix: sparse.csr_matrix):
        """Compute different variables from labels_."""
        if self.return_membership or self.return_aggregate:
            input_matrix = input_matrix.astype(float)
            if not self.bipartite:
                membership = membership_matrix(self.labels_)
                if self.return_membership:
                    self.membership_ = normalize(input_matrix.dot(membership))
                if self.return_aggregate:
                    self.aggregate_ = sparse.csr_matrix(membership.T.dot(input_matrix.dot(membership)))
            else:
                if self.labels_col_ is None:
                    n_labels = max(self.labels_) + 1
                    membership_row = membership_matrix(self.labels_, n_labels=n_labels)
                    membership_col = normalize(input_matrix.T.dot(membership_row))
                else:
                    n_labels = max(max(self.labels_row_), max(self.labels_col_)) + 1
                    membership_row = membership_matrix(self.labels_row_, n_labels=n_labels)
                    membership_col = membership_matrix(self.labels_col_, n_labels=n_labels)
                if self.return_membership:
                    self.membership_row_ = normalize(input_matrix.dot(membership_col))
                    self.membership_col_ = normalize(input_matrix.T.dot(membership_row))
                    self.membership_ = self.membership_row_
                if self.return_aggregate:
                    aggregate_ = sparse.csr_matrix(membership_row.T.dot(input_matrix))
                    aggregate_ = aggregate_.dot(membership_col)
                    self.aggregate_ = aggregate_

        return self
