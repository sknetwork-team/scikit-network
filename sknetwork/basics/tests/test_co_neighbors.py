#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.basics import co_neighbors_graph, CoNeighbors
from sknetwork.data import movie_actor
from sknetwork.linalg.normalization import normalize


class TestCoNeighbors(unittest.TestCase):

    def setUp(self):
        """Simple biadjacency for tests."""
        self.biadjacency = movie_actor()

    def test_exact(self):
        n = self.biadjacency.shape[0]
        adjacency = co_neighbors_graph(self.biadjacency, method='exact', normalized=False)
        self.assertEqual(adjacency.shape, (n, n))
        adjacency = co_neighbors_graph(self.biadjacency, method='exact')
        self.assertEqual(adjacency.shape, (n, n))

        operator = CoNeighbors(self.biadjacency)
        x = np.random.randn(n)
        y1 = adjacency.dot(x)
        y2 = operator.dot(x)
        self.assertAlmostEqual(np.linalg.norm(y1 - y2), 0)

    def test_knn(self):
        n = self.biadjacency.shape[0]
        adjacency = co_neighbors_graph(self.biadjacency, method='knn')
        self.assertEqual(adjacency.shape, (n, n))
        adjacency = co_neighbors_graph(self.biadjacency, method='knn', normalized=False)
        self.assertEqual(adjacency.shape, (n, n))

    def test_operator(self):
        operator = CoNeighbors(self.biadjacency)
        transition = normalize(operator)
        x = transition.dot(np.ones(transition.shape[1]))

        self.assertAlmostEqual(np.linalg.norm(x - np.ones(operator.shape[0])), 0)
        operator.astype(np.float)
        operator.right_sparse_dot(sparse.eye(operator.shape[1], format='csr'))

        operator1 = CoNeighbors(self.biadjacency, normalized=False)
        operator2 = CoNeighbors(self.biadjacency, normalized=False)
        x = np.random.randn(operator.shape[1])
        x1 = (-operator1).dot(x)
        x2 = (operator2 * -1).dot(x)
        self.assertAlmostEqual(np.linalg.norm(x1 - x2), 0)
