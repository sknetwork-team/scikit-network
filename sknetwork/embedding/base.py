#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC

import numpy as np
from scipy import sparse

from sknetwork.topology.structure import is_connected
from sknetwork.utils.base import Algorithm


class BaseEmbedding(Algorithm, ABC):
    """Base class for embedding algorithms."""

    def __init__(self):
        self._init_vars()

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit to data and return the embedding. Same parameters as the ``fit`` method.

        Returns
        -------
        embedding : np.ndarray
            Embedding.
        """
        self.fit(*args, **kwargs)
        return self.embedding_

    def _check_fitted(self):
        if self.embedding_ is None:
            raise ValueError("This embedding instance is not fitted yet."
                             " Call 'fit' with appropriate arguments before using this method.")
        else:
            return self

    @staticmethod
    def _get_regularization(regularization: float, adjacency: sparse.csr_matrix) -> float:
        """Set proper regularization depending on graph connectivity."""
        if regularization < 0:
            if is_connected(adjacency, connection='strong'):
                regularization = 0
            else:
                regularization = np.abs(regularization)
        return regularization

    def _init_vars(self):
        self.embedding_ = None
        self.embedding_row_ = None
        self.embedding_col_ = None

    def _split_vars(self, shape):
        """Split labels_ into labels_row_ and labels_col_"""
        n_row = shape[0]
        self.embedding_row_ = self.embedding_[:n_row]
        self.embedding_col_ = self.embedding_[n_row:]
        self.embedding_ = self.embedding_row_
        return self
