#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from scipy import sparse
import numpy as np

from sknetwork.utils import KNeighborsTransformer


class TestKNeighbors(unittest.TestCase):

    def test_basics(self):
        self.x = np.array([[-2, -1], [-2, 1], [2, 1], [2, -1]])
        knn = KNeighborsTransformer(n_neighbors=1)
        knn.fit(self.x)
        gt = np.zeros(16).reshape((4, 4))
        gt[0, 1] = 1
        gt[2, 3] = 1
        gt = sparse.csr_matrix(gt + gt.T)
        self.assertAlmostEqual(sparse.linalg.norm(gt - knn.adjacency_), 0)
