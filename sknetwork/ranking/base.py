#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from sknetwork.utils.base import Algorithm


class BaseRanking(Algorithm):
    """Base class for ranking algorithms."""

    def __init__(self):
        self.score_ = None
