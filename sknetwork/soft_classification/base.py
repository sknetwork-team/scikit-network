#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC

from sknetwork.utils.base import Algorithm


class BaseSoftClassifier(Algorithm, ABC):
    """Base class for soft clustering algorithms."""

    def __init__(self):
        self.membership_ = None

    def fit_transform(self, *args, **kwargs):
        """Fit algorithm to the data and returns the membership matrix."""
        self.fit(*args, **kwargs)
        return self.membership_
