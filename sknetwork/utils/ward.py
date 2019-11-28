#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy.cluster.hierarchy import ward

from sknetwork.utils.base import Algorithm


class Ward(Algorithm):
    """Standard Ward hierarchical clustering based on SciPy.

    Attributes
    ----------
    dendrogram_ : numpy array of shape (total number of nodes - 1, 4)
        Dendrogram.

    References
    ----------
    Murtagh, F., & Contreras, P. (2012). Algorithms for hierarchical clustering: an overview.
    Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 2(1), 86-97.

    """

    def __init__(self):
        self.dendrogram_ = None

    def fit(self, x: np.ndarray) -> 'Ward':
        """Apply algorithm to a dense matrix.

        Parameters
        ----------
        x:
            Data to cluster.

        Returns
        -------
        self: :class:`Ward`

        """
        self.dendrogram_ = ward(x)

        return self
