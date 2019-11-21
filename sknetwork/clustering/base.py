#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC

from sknetwork.utils.base import Algorithm


class BaseClustering(Algorithm, ABC):
    """Base class for clustering algorithms."""

    def __init__(self):
        self.labels_ = None

    def fit_transform(self, adjacency):
        """Fit algorithm to the data and returns the labels."""
        self.fit(adjacency)
        return self.labels_
