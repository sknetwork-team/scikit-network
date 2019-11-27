#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from sknetwork.basics import co_neighbors_graph
from sknetwork.data import movie_actor


# noinspection PyMissingOrEmptyDocstring
class TestCoNeighbors(unittest.TestCase):

    def setUp(self):
        self.bipartite = movie_actor(return_labels=False)

    def test_exact(self):
        n = self.bipartite.shape[0]
        co_neighbors = co_neighbors_graph(self.bipartite, method='exact')
        self.assertEqual(co_neighbors.shape, (n, n))
        co_neighbors = co_neighbors_graph(self.bipartite, method='exact', normalized=False)
        self.assertEqual(co_neighbors.shape, (n, n))

    def test_knn(self):
        n = self.bipartite.shape[0]
        co_neighbors = co_neighbors_graph(self.bipartite, method='knn')
        self.assertEqual(co_neighbors.shape, (n, n))
        co_neighbors = co_neighbors_graph(self.bipartite, method='knn', normalized=False)
        self.assertEqual(co_neighbors.shape, (n, n))
