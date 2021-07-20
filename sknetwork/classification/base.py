#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC

import numpy as np

from sknetwork.utils.base import Algorithm


class BaseClassifier(Algorithm, ABC):
    """Base class for classifiers.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each row.
    membership_ : sparse.csr_matrix
        Membership matrix of rows (soft classification, labels on columns).
    """

    def __init__(self):
        self.labels_ = None
        self.membership_ = None
        self.labels_row_ = None
        self.labels_col_ = None
        self.membership_row_ = None
        self.membership_col_ = None

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
        self.fit(*args, **kwargs)
        return self.labels_

    def score(self, label: int):
        """Classification scores for a given label.

        Parameters
        ----------
        label : int
            The label index of the class.

        Returns
        -------
        scores : np.ndarray
            Classification scores of shape (number of nodes,).
        """
        if self.membership_ is None:
            raise ValueError("The fit method should be called first.")
        return self.membership_[:, label].toarray().ravel()

    def _split_vars(self, shape: tuple):
        n_row = shape[0]
        self.labels_row_ = self.labels_[:n_row]
        self.labels_col_ = self.labels_[n_row:]
        self.labels_ = self.labels_row_
        self.membership_row_ = self.membership_[:n_row]
        self.membership_col_ = self.membership_[n_row:]
        self.membership_ = self.membership_row_
        return self
