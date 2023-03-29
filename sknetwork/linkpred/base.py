#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in March 2022
@author: Thomas Bonald <thomas.bonald@telecom-paris.fr>
"""
from abc import ABC

import numpy as np

from sknetwork.utils.base import Algorithm


class BaseLinker(Algorithm, ABC):
    """Base class for link prediction.

    Attributes
    ----------
    links_: sparse.csr_matrix
        Link matrix.
    """

    def __init__(self):
        self.links_ = None

    def fit_predict(self, *args, **kwargs) -> np.ndarray:
        """Fit algorithm to data and return the links. Same parameters as the ``fit`` method.

        Returns
        -------
        links_ : sparse.csr_matrix
            Link matrix.
        """
        self.fit(*args, **kwargs)
        return self.links_
