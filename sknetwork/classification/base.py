#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in November 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from abc import ABC

import numpy as np
from scipy import sparse

from sknetwork.utils.base import Algorithm


class BaseClassifier(Algorithm, ABC):
    """Base class for classifiers.

    Attributes
    ----------
    bipartite : bool
        If ``True``, the fitted graph is bipartite.
    labels_ : np.ndarray, shape (n_labels,)
        Label of each node.
    membership_ : sparse.csr_matrix, shape (n_row, n_labels)
        Membership matrix (soft classification).
    labels_row_ , labels_col_ : np.ndarray
        Label of rows and columns (for bipartite graphs).
    membership_row_, membership_col_ : sparse.csr_matrix, shapes (n_row, n_labels) and (n_col, n_labels)
        Membership matrices of rows and columns (for bipartite graphs).
    """

    def __init__(self):
        self.bipartite = None
        self.labels_ = None
        self.membership_ = None
        self.labels_row_ = None
        self.labels_col_ = None
        self.membership_row_ = None
        self.membership_col_ = None

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

    def fit_transform(self, *args, **kwargs) -> sparse.csr_matrix:
        """Fit algorithm to the data and return the probability distribution over labels in sparse format.
        Same parameters as the ``fit`` method.

        Returns
        -------
        membership : sparse.csr_matrix
            Probability of each label.
        """
        self.fit(*args, **kwargs)
        return self.transform()

    def _split_vars(self, shape: tuple):
        """Split variables for bipartite graphs."""
        if self.bipartite:
            n_row = shape[0]
            self.labels_row_ = self.labels_[:n_row]
            self.labels_col_ = self.labels_[n_row:]
            self.labels_ = self.labels_row_
            self.membership_row_ = self.membership_[:n_row]
            self.membership_col_ = self.membership_[n_row:]
            self.membership_ = self.membership_row_
        else:
            self.labels_row_ = self.labels_
            self.labels_col_ = self.labels_
            self.membership_row_ = self.membership_
            self.membership_col_ = self.membership_
        return self
