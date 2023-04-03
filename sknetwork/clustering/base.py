#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from abc import ABC

import numpy as np
from scipy import sparse

from sknetwork.linalg.normalization import normalize
from sknetwork.utils.base import Algorithm
from sknetwork.utils.membership import get_membership


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

    def predict(self, columns=False) -> np.ndarray:
        """Return the labels predicted by the algorithm.

        Parameters
        ----------
        columns : bool
            If ``True``, return the prediction for columns.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
        if columns:
            return self.labels_col_
        return self.labels_

    def fit_predict(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
        self.fit(*args, **kwargs)
        return self.predict()

    def predict_proba(self, columns=False) -> np.ndarray:
        """Return the probability distribution over labels as predicted by the algorithm.

        Parameters
        ----------
        columns : bool
            If ``True``, return the prediction for columns.

        Returns
        -------
        probs : np.ndarray
            Probability distribution over labels.
        """
        if columns:
            return self.membership_col_.toarray()
        return self.membership_.toarray()

    def fit_predict_proba(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the probability distribution over labels.
        Same parameters as the ``fit`` method.

        Returns
        -------
        probs : np.ndarray
            Probability of each label.
        """
        self.fit(*args, **kwargs)
        return self.predict_proba()

    def transform(self, columns=False) -> sparse.csr_matrix:
        """Return the probability distribution over labels in sparse format.

        Parameters
        ----------
        columns : bool
            If ``True``, return the prediction for columns.

        Returns
        -------
        membership : sparse.csr_matrix
            Probability distribution over labels (aka membership matrix).
        """
        if columns:
            return self.membership_col_
        return self.membership_

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the membership matrix. Same parameters as the ``fit`` method.

        Returns
        -------
        membership : np.ndarray
            Membership matrix (distribution over clusters).
        """
        self.fit(*args, **kwargs)
        return self.transform()

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
                membership = get_membership(self.labels_)
                if self.return_membership:
                    self.membership_ = normalize(input_matrix.dot(membership))
                if self.return_aggregate:
                    self.aggregate_ = sparse.csr_matrix(membership.T.dot(input_matrix.dot(membership)))
            else:
                if self.labels_col_ is None:
                    n_labels = max(self.labels_) + 1
                    membership_row = get_membership(self.labels_, n_labels=n_labels)
                    membership_col = normalize(input_matrix.T.dot(membership_row))
                else:
                    n_labels = max(max(self.labels_row_), max(self.labels_col_)) + 1
                    membership_row = get_membership(self.labels_row_, n_labels=n_labels)
                    membership_col = get_membership(self.labels_col_, n_labels=n_labels)
                if self.return_membership:
                    self.membership_row_ = normalize(input_matrix.dot(membership_col))
                    self.membership_col_ = normalize(input_matrix.T.dot(membership_row))
                    self.membership_ = self.membership_row_
                if self.return_aggregate:
                    aggregate_ = sparse.csr_matrix(membership_row.T.dot(input_matrix))
                    aggregate_ = aggregate_.dot(membership_col)
                    self.aggregate_ = aggregate_

        return self
