#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from sknetwork.utils.base import Algorithm


class BaseHierarchy(Algorithm):
    """Base class for hierarchical clustering algorithms."""

    def __init__(self):
        self.dendrogram_ = None
