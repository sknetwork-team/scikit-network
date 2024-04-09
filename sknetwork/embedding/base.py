#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in November 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from abc import ABC
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.topology.structure import is_connected
from sknetwork.base import Algorithm


class BaseEmbedding(Algorithm, ABC):
    """Base class for embedding algorithms.

    Attributes
    ----------
    embedding_ : array, shape = (n, n_components)
        Embedding of the nodes.
    embedding_row_ : array, shape = (n_row, n_components)
        Embedding of the rows, for bipartite graphs.
    embedding_col_ : array, shape = (n_col, n_components)
        Embedding of the columns, for bipartite graphs.
    """
    def __init__(self):
        self._init_vars()

    def transform(self) -> np.ndarray:
        """Return the embedding.

        Returns
        -------
        embedding : np.ndarray
            Embedding.
        """
        return self.embedding_

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit to data and return the embedding. Same parameters as the ``fit`` method.

        Returns
        -------
        embedding : np.ndarray
            Embedding.
        """
        self.fit(*args, **kwargs)
        return self.embedding_

    def predict(self, columns: bool = False) -> np.ndarray:
        """Return the embedding of nodes.

        Parameters
        ----------
        columns : bool
            If ``True``, return the prediction for columns.

        Returns
        -------
        embedding_ : np.ndarray
            Embedding of the nodes.
        """
        if columns:
            return self.embedding_col_
        return self.embedding_

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

    def _check_fitted(self):
        return self.embedding_ is not None

    def _split_vars(self, shape):
        """Split labels_ into labels_row_ and labels_col_"""
        n_row = shape[0]
        self.embedding_row_ = self.embedding_[:n_row]
        self.embedding_col_ = self.embedding_[n_row:]
        self.embedding_ = self.embedding_row_
        return self
