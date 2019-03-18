#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for louvain.py"""


import unittest
from sknetwork.clustering import Louvain
from sknetwork.toy_graphs import karate_club_graph
from scipy.sparse import identity


class TestLouvainClustering(unittest.TestCase):

    def setUp(self):
        self.louvain = Louvain()

    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.louvain.fit(identity(1))

        with self.assertRaises(TypeError):
            self.louvain.fit(identity(2, format='csr'), node_weights=1)

    def test_unknown_options(self):
        with self.assertRaises(ValueError):
            self.louvain.fit(identity(2, format='csr'), node_weights='unknown')

    def test_single_node_graph(self):
        self.assertEqual(self.louvain.fit(identity(1, format='csr')).labels_, [0])

    def test_karate_graph(self):
        self.assertEqual(self.louvain.fit(karate_club_graph()).labels_.shape, (34,))
