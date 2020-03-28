#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC

from sknetwork.utils.base import Algorithm


class BaseHierarchy(Algorithm, ABC):
    """Base class for hierarchical clustering algorithms."""

    def __init__(self):
        self.dendrogram_ = None

    def fit_transform(self, *args, **kwargs):
        """Fit algorithm to the data and returns the dendrogram. Use the same inputs as the ``fit`` method."""
        self.fit(*args, **kwargs)
        return self.dendrogram_
