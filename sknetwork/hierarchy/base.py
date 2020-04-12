#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC

import numpy as np

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
