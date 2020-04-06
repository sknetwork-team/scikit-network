#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from sknetwork.basics import co_neighbors_graph
from sknetwork.data import movie_actor


class TestCoNeighbors(unittest.TestCase):

    def setUp(self):
        """Simple biadjacency for tests."""
        self.biadjacency = movie_actor()

    def test_exact(self):
        n = self.biadjacency.shape[0]
        co_neighbors = co_neighbors_graph(self.biadjacency, method='exact')
        self.assertEqual(co_neighbors.shape, (n, n))
        co_neighbors = co_neighbors_graph(self.biadjacency, method='exact', normalize=False)
        self.assertEqual(co_neighbors.shape, (n, n))

    def test_knn(self):
        n = self.biadjacency.shape[0]
        co_neighbors = co_neighbors_graph(self.biadjacency, method='knn')
        self.assertEqual(co_neighbors.shape, (n, n))
        co_neighbors = co_neighbors_graph(self.biadjacency, method='knn', normalize=False)
        self.assertEqual(co_neighbors.shape, (n, n))
