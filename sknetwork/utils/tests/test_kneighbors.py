#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm

from sknetwork.utils import KNeighborsTransformer, FWKNeighborsTransformer


class TestKNeighbors(unittest.TestCase):

    def test_basics(self):
        x = np.array([[-2, -1], [-2, 1], [2, 1], [2, -1]])
        knn = KNeighborsTransformer(n_neighbors=1)
        knn.fit(x)
        gt = np.zeros(16).reshape((4, 4))
        gt[0, 1] = 1
        gt[2, 3] = 1
        gt = sparse.csr_matrix(gt + gt.T)
        self.assertAlmostEqual(norm(gt - knn.adjacency_), 0)

    def test_fwknn(self):
        x = np.array([(0, 4), (1, 0), (2, 1), (3, 2), (4, 3)])
        fwknn = FWKNeighborsTransformer(n_neighbors=1)
        adj = fwknn.fit_transform(x)
        self.assertEqual(adj.shape, (5, 5))
