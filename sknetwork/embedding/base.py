#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC

from sknetwork.utils.base import Algorithm


class BaseEmbedding(Algorithm, ABC):
    """Base class for embedding algorithms."""

    def __init__(self):
        self.embedding_ = None

    def fit_transform(self, *args, **kwargs):
        """Fit algorithm to the data and return the embedding of the rows.
        Uses the same inputs as this class ``fit`` method."""
        self.fit(*args, **kwargs)
        return self.embedding_
