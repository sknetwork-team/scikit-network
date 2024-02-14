#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in November 2019
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from abc import ABC

import numpy as np

from sknetwork.hierarchy.postprocess import split_dendrogram
from sknetwork.base import Algorithm


class BaseHierarchy(Algorithm, ABC):
    """Base class for hierarchical clustering algorithms.
    Attributes
    ----------
    dendrogram_ :
        Dendrogram of the graph.
    dendrogram_row_ :
        Dendrogram for the rows, for bipartite graphs.
    dendrogram_col_ :
        Dendrogram for the columns, for bipartite graphs.
    dendrogram_full_ :
        Dendrogram for both rows and columns, indexed in this order, for bipartite graphs.
    """

    def __init__(self):
        self._init_vars()

    def predict(self, columns: bool = False) -> np.ndarray:
        """Return the dendrogram predicted by the algorithm.

        Parameters
        ----------
        columns : bool
            If ``True``, return the prediction for columns.

        Returns
        -------
        dendrogram : np.ndarray
            Dendrogram.
        """
        if columns:
            return self.dendrogram_col_
        return self.dendrogram_

    def transform(self) -> np.ndarray:
        """Return the dendrogram predicted by the algorithm.

        Returns
        -------
        dendrogram : np.ndarray
            Dendrogram.
        """
        return self.dendrogram_

    def fit_predict(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to data and return the dendrogram. Same parameters as the ``fit`` method.

        Returns
        -------
        dendrogram : np.ndarray
            Dendrogram.
        """
        self.fit(*args, **kwargs)
        return self.dendrogram_

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to data and return the dendrogram. Alias for ``fit_predict``.
        Same parameters as the ``fit`` method.

        Returns
        -------
        dendrogram : np.ndarray
            Dendrogram.
        """
        self.fit(*args, **kwargs)
        return self.dendrogram_

    def _init_vars(self):
        """Init variables."""
        self.dendrogram_ = None
        self.dendrogram_row_ = None
        self.dendrogram_col_ = None
        self.dendrogram_full_ = None

    def _split_vars(self, shape):
        """Split variables."""
        dendrogram_row, dendrogram_col = split_dendrogram(self.dendrogram_, shape)
        self.dendrogram_full_ = self.dendrogram_
        self.dendrogram_ = dendrogram_row
        self.dendrogram_row_ = dendrogram_row
        self.dendrogram_col_ = dendrogram_col
        return self
