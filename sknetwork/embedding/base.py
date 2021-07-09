#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from abc import ABC
from typing import Union, Tuple

import numpy as np
from scipy import sparse

from sknetwork.topology.structure import is_connected
from sknetwork.utils.base import Algorithm
from sknetwork.utils.check import check_format, is_square, is_symmetric
from sknetwork.utils.format import bipartite2undirected


class BaseEmbedding(Algorithm, ABC):
    """Base class for embedding algorithms."""

    def __init__(self):
        self.embedding_ = None
        self.embedding_row_ = None
        self.embedding_col_ = None

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
    def _check_input(input_matrix: Union[sparse.csr_matrix, np.ndarray], symmetric: bool = True)\
            -> Tuple[sparse.csr_matrix, bool, tuple]:
        """Check the input matrix and return a proper adjacency matrix."""
        input_matrix = check_format(input_matrix)
        input_shape = input_matrix.shape
        if not is_square(input_matrix) or (symmetric and not is_symmetric(input_matrix)):
            bipartite = True
            adjacency = bipartite2undirected(input_matrix)
        else:
            bipartite = False
            adjacency = input_matrix
        return adjacency, bipartite, input_shape

    @staticmethod
    def _get_regularization(regularization: float, adjacency: sparse.csr_matrix) -> float:
        """Set proper regularization depending on graph connectivity."""
        if regularization < 0:
            if is_connected(adjacency, connection='strong'):
                regularization = 0
            else:
                regularization = np.abs(regularization)
        return regularization

    def _split_vars(self, n_row):
        """Split labels_ into labels_row_ and labels_col_"""
        self.embedding_row_ = self.embedding_[:n_row]
        self.embedding_col_ = self.embedding_[n_row:]
        self.embedding_ = self.embedding_row_
        return self


class BaseBiEmbedding(BaseEmbedding, ABC):
    """Base class for embedding algorithms."""

    def __init__(self):
        super(BaseBiEmbedding, self).__init__()
        self.embedding_row_ = None
        self.embedding_col_ = None

    def _split_vars(self, n_row):
        """Split labels_ into labels_row_ and labels_col_"""
        self.embedding_row_ = self.embedding_[:n_row]
        self.embedding_col_ = self.embedding_[n_row:]
        self.embedding_ = self.embedding_row_
        return self
