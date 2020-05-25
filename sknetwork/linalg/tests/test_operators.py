#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
import unittest

import numpy as np
from scipy import sparse

from sknetwork.data import movie_actor
from sknetwork.linalg import CoNeighborOperator, normalize


class TestOperators(unittest.TestCase):

    def test_coneighbors(self):
        biadjacency = movie_actor(metadata=False)
        operator = CoNeighborOperator(biadjacency)
        transition = normalize(operator)
        x = transition.dot(np.ones(transition.shape[1]))

        self.assertAlmostEqual(np.linalg.norm(x - np.ones(operator.shape[0])), 0)
        operator.astype(np.float)
        operator.right_sparse_dot(sparse.eye(operator.shape[1], format='csr'))

        operator1 = CoNeighborOperator(biadjacency, normalized=False)
        operator2 = CoNeighborOperator(biadjacency, normalized=False)
        x = np.random.randn(operator.shape[1])
        x1 = (-operator1).dot(x)
        x2 = (operator2 * -1).dot(x)
        x3 = operator1.T.dot(x)
        self.assertAlmostEqual(np.linalg.norm(x1 - x2), 0)
        self.assertAlmostEqual(np.linalg.norm(x2 - x3), 0)
