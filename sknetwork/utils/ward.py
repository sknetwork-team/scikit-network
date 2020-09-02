#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
import numpy as np
from scipy.cluster.hierarchy import ward

from sknetwork.utils.base import Algorithm


class WardDense(Algorithm):
    """Hierarchical clustering by the Ward method based on SciPy.

    Attributes
    ----------
    dendrogram_ : np.ndarray (n - 1, 4)
        Dendrogram.

    References
    ----------
    * Ward, J. H., Jr. (1963). Hierarchical grouping to optimize an objective function.
      Journal of the American Statistical Association, 58, 236–244.

    * Murtagh, F., & Contreras, P. (2012). Algorithms for hierarchical clustering: an overview.
      Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 2(1), 86-97.
    """
    def __init__(self):
        self.dendrogram_ = None

    def fit(self, x: np.ndarray) -> 'WardDense':
        """Apply algorithm to a dense matrix.

        Parameters
        ----------
        x:
            Data to cluster.

        Returns
        -------
        self: :class:`WardDense`
        """
        self.dendrogram_ = ward(x)
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply algorithm to a dense matrix and return the dendrogram.

        Parameters
        ----------
        x:
            Data to cluster.

        Returns
        -------
        dendrogram: np.ndarray
        """
        self.fit(x)
        return self.dendrogram_
