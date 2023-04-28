#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 28, 2019
@author: Quentin Lutz <qlutz@enst.fr>
"""
import inspect

from skbase.base import BaseEstimator


class Algorithm(BaseEstimator):
    """Base class for all algorithms."""

    def fit(self, *args, **kwargs):
        """Fit algorithm to data."""
        raise NotImplementedError
