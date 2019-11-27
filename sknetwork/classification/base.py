#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC

from sknetwork.utils.base import Algorithm


class BaseClassifier(Algorithm, ABC):
    """Base class for classifiers"""

    def __init__(self):
        self.labels_ = None

    def fit_transform(self, *args, **kwargs):
        """Fit algorithm to the data and return the labels. Uses the same inputs as this class ``fit`` method."""
        self.fit(*args, **kwargs)
        return self.labels_
