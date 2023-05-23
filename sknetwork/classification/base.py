#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in November 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from abc import ABC

import numpy as np
from scipy import sparse

from sknetwork.base import Algorithm


class BaseClassifier(Algorithm, ABC):
    """Base class for classifiers.

    Attributes
    ----------
    bipartite : bool
        If ``True``, the fitted graph is bipartite.
    labels_ : np.ndarray, shape (n_labels,)
        Labels of nodes.
    probs_ : sparse.csr_matrix, shape (n_row, n_labels)
        Probability distribution over labels (soft classification).
    labels_row_ , labels_col_ : np.ndarray
        Labels of rows and columns (for bipartite graphs).
    probs_row_, probs_col_ : sparse.csr_matrix, shapes (n_row, n_labels) and (n_col, n_labels)
        Probability distributions over labels for rows and columns (for bipartite graphs).
    """

    def __init__(self):
        self.bipartite = None
        self.labels_ = None
        self.probs_ = None
        self.labels_row_ = None
        self.labels_col_ = None
        self.probs_row_ = None
        self.probs_col_ = None

    def predict(self, columns: bool = False) -> np.ndarray:
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
            return self.probs_col_.toarray()
        return self.probs_.toarray()

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
        probs : sparse.csr_matrix
            Probability distribution over labels.
        """
        if columns:
            return self.probs_col_
        return self.probs_

    def fit_transform(self, *args, **kwargs) -> sparse.csr_matrix:
        """Fit algorithm to the data and return the probability distribution over labels in sparse format.
        Same parameters as the ``fit`` method.

        Returns
        -------
        probs : sparse.csr_matrix
            Probability distribution over labels.
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
            self.probs_row_ = self.probs_[:n_row]
            self.probs_col_ = self.probs_[n_row:]
            self.probs_ = self.probs_row_
        else:
            self.labels_row_ = self.labels_
            self.labels_col_ = self.labels_
            self.probs_row_ = self.probs_
            self.probs_col_ = self.probs_
        return self
