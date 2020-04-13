#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC

import numpy as np

from sknetwork.utils.base import Algorithm


class BaseClustering(Algorithm, ABC):
    """Base class for clustering algorithms.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.
    membership_ : sparse.csr_matrix
        Membership matrix.
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
