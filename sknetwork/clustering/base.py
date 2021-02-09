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
        Label of each node.
    membership_ : sparse.csr_matrix
        Membership matrix.
    adjacency_ : sparse.csr_matrix
        Adjacency matrix between clusters.
    """
    def __init__(self, sort_clusters: bool = True, return_membership: bool = False, return_aggregate: bool = False):
        self.sort_clusters = sort_clusters
        self.return_membership = return_membership
        self.return_aggregate = return_aggregate

        self.labels_ = None
        self.membership_ = None
        self.adjacency_ = None

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
        self.fit(*args, **kwargs)
        return self.labels_

    def _secondary_outputs(self, adjacency):
        """Compute different variables from labels_."""
        if self.return_membership or self.return_aggregate:
            if np.issubdtype(adjacency.data.dtype, np.bool_):
                adjacency = adjacency.astype(float)
            membership = membership_matrix(self.labels_)
            if self.return_membership:
                self.membership_ = normalize(adjacency.dot(membership))
            if self.return_aggregate:
                self.adjacency_ = sparse.csr_matrix(membership.T.dot(adjacency.dot(membership)))
        return self


class BaseBiClustering(BaseClustering, ABC):
    """Base class for clustering algorithms.

    Attributes
    ----------
    labels_ : np.ndarray
        Labels of the rows.
    labels_row_ : np.ndarray
        Labels of the rows (copy of **labels_**).
    labels_col_ : np.ndarray
        Labels of the columns.
    membership_ : sparse.csr_matrix
        Membership matrix of the rows (copy of **membership_**).
    membership_row_ : sparse.csr_matrix
        Membership matrix of the rows (copy of **membership_**).
    membership_col_ : sparse.csr_matrix
        Membership matrix of the columns. Only valid if **cluster_both** = `True`.
    biadjacency_ : sparse.csr_matrix
        Biadjacency matrix of the aggregate graph between clusters.
    """

    def __init__(self, sort_clusters: bool = True, return_membership: bool = False, return_aggregate: bool = False):
        super(BaseBiClustering, self).__init__(sort_clusters=sort_clusters, return_membership=return_membership,
                                               return_aggregate=return_aggregate)

        self.labels_row_ = None
        self.labels_col_ = None
        self.membership_row_ = None
        self.membership_col_ = None
        self.biadjacency_ = None

    def _split_vars(self, n_row):
        """Split labels_ into labels_row_ and labels_col_"""
        self.labels_row_ = self.labels_[:n_row]
        self.labels_col_ = self.labels_[n_row:]
        self.labels_ = self.labels_row_
        return self

    def _secondary_outputs(self, biadjacency):
        """Compute different variables from labels_."""
        if self.return_membership or self.return_aggregate:
            if np.issubdtype(biadjacency.data.dtype, np.bool_):
                biadjacency = biadjacency.astype(float)

        if self.return_membership:
            n_labels = max(max(self.labels_row_), max(self.labels_col_)) + 1
            membership_row = membership_matrix(self.labels_row_, n_labels=n_labels)
            membership_col = membership_matrix(self.labels_col_, n_labels=n_labels)
            self.membership_row_ = normalize(biadjacency.dot(membership_col))
            self.membership_col_ = normalize(biadjacency.T.dot(membership_row))
            self.membership_ = self.membership_row_

        if self.return_aggregate:
            n_labels = max(max(self.labels_row_), max(self.labels_col_)) + 1
            membership_row = membership_matrix(self.labels_row_, n_labels=n_labels)
            membership_col = membership_matrix(self.labels_col_, n_labels=n_labels)
            biadjacency_ = sparse.csr_matrix(membership_row.T.dot(biadjacency))
            biadjacency_ = biadjacency_.dot(membership_col)
            self.biadjacency_ = biadjacency_
        return self
