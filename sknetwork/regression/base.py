#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2022
@author: Thomas Bonald <bonald@enst.fr>
"""
from abc import ABC

import numpy as np

from sknetwork.base import Algorithm


class BaseRegressor(Algorithm, ABC):
    """Base class for regression algorithms.

    Attributes
    ----------
    values_ : np.ndarray
        Value of each node.
    values_row_: np.ndarray
        Values of rows, for bipartite graphs.
    values_col_: np.ndarray
        Values of columns, for bipartite graphs.
    """
    def __init__(self):
        self.values_ = None

    def predict(self, columns: bool = False) -> np.ndarray:
        """Return the values predicted by the algorithm.

        Parameters
        ----------
        columns : bool
            If ``True``, return the prediction for columns.

        Returns
        -------
        values : np.ndarray
            Values.
        """
        if columns:
            return self.values_col_
        return self.values_

    def fit_predict(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to data and return the values. Same parameters as the ``fit`` method.

        Returns
        -------
        values : np.ndarray
            Values.
        """
        self.fit(*args, **kwargs)
        return self.values_

    def _split_vars(self, shape):
        n_row = shape[0]
        self.values_row_ = self.values_[:n_row]
        self.values_col_ = self.values_[n_row:]
        self.values_ = self.values_row_
