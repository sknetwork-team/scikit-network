#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from sknetwork.utils.base import Algorithm


class BaseClassifier(Algorithm):
    """Base class for classifiers"""

    def __init__(self):
        self.labels_ = None
