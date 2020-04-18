#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC

import numpy as np

from sknetwork.hierarchy.postprocess import split_dendrogram
from sknetwork.utils.base import Algorithm


class BaseHierarchy(Algorithm, ABC):
    """Base class for hierarchical clustering algorithms."""

    def __init__(self):
        self.dendrogram_ = None

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to data and return the dendrogram. Same parameters as the ``fit`` method.

        Returns
        -------
        dendrogram : np.ndarray
            Dendrogram.
        """
        self.fit(*args, **kwargs)
        return self.dendrogram_


class BaseBiHierarchy(BaseHierarchy, ABC):
    """Base class for hierarchical clustering algorithms."""

    def __init__(self):
        super(BaseBiHierarchy, self).__init__()
        self.dendrogram_row_ = None
        self.dendrogram_col_ = None
        self.dendrogram_full_ = None

    def _split_vars(self, shape):
        dendrogram_row, dendrogram_col = split_dendrogram(self.dendrogram_, shape)

        self.dendrogram_full_ = self.dendrogram_
        self.dendrogram_ = dendrogram_row
        self.dendrogram_row_ = dendrogram_row
        self.dendrogram_col_ = dendrogram_col

        return self
