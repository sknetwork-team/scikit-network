#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC

from sknetwork.utils.base import Algorithm


class BaseClassifier(Algorithm, ABC):
    """Base class for classifiers."""

    def __init__(self):
        self.labels_ = None
        self.membership_ = None

    def fit_transform(self, *args, **kwargs):
        """Fit algorithm to the data and return the labels. Use the same inputs as the ``fit`` method."""
        self.fit(*args, **kwargs)
        return self.labels_


class BaseBiClassifier(BaseClassifier, ABC):
    """Base class for classifiers."""

    def __init__(self):
        super(BaseBiClassifier, self).__init__()

        self.labels_row_ = None
        self.labels_col_ = None
        self.membership_row_ = None
        self.membership_col_ = None
