#!/usr/bin/env python3
# coding: utf-8
"""
Created on Jun 2020
@author: Victor Manach <victor.manach@telecom-paris.fr>
@author: Rémi Jaylet <remi.jaylet@telecom-paris.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.embedding.base import BaseEmbedding
from sknetwork.utils import directed2undirected
from sknetwork.utils.check import check_format, is_symmetric, check_square


class Force_atlas(BaseEmbedding):

    def __init__(self, strength: float = None, n_iter: int = 50, tol: float = 1e-4):
        super(Force_atlas, self).__init__()
        self.strength = strength
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], n_iter: Optional[int] = None) -> 'Spring':
        adjacency = check_format(adjacency)
        check_square(adjacency)
        if not is_symmetric(adjacency):
            adjacency = directed2undirected(adjacency)
        n = adjacency.shape[0]

        position = np.zeros((n, 2))

