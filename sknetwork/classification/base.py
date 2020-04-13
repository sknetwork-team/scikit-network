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

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to the data and return the labels. Same parameters as the ``fit`` method.

        Returns
        -------
        labels : np.ndarray
            Labels.
        """
        self.fit(*args, **kwargs)
        return self.labels_


class BaseBiClassifier(BaseClassifier, ABC):
    """Base class for classifiers on bigraphs.

    Attributes
    ----------
    labels_row_ : np.ndarray
        Label of each row (copy of **labels_**).
    labels_col_ : np.ndarray
        Label of each column.
    membership_row_ : sparse.csr_matrix
        Membership matrix of rows (copy of **membership_**).
    membership_col_ : sparse.csr_matrix
        Membership matrix of columns.
    """

    def __init__(self):
        super(BaseBiClassifier, self).__init__()

        self.labels_row_ = None
        self.labels_col_ = None
        self.membership_row_ = None
        self.membership_col_ = None
