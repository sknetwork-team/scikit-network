#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for louvain.py"""


import unittest
from sknetwork.clustering import *
from sknetwork.toy_graphs import *
from scipy.sparse import identity


class TestLouvainClustering(unittest.TestCase):

    def setUp(self):
        self.louvain = Louvain()
        self.louvain_high_resolution = Louvain(GreedyModularity(engine='python', resolution=2))
        self.louvain_null_resolution = Louvain(GreedyModularity(engine='python', resolution=0))
        if is_numba_available:
            self.louvain_numba = Louvain(GreedyModularity(engine='numba'))
        else:
            with self.assertRaises(ValueError):
                Louvain(GreedyModularity(engine='numba'))
        self.louvain_shuffle_first = Louvain(GreedyModularity(engine='python'), shuffle_nodes=True, random_state=0)
        self.louvain_shuffle_second = Louvain(GreedyModularity(engine='python'), shuffle_nodes=True, random_state=123)
        self.karate_club_graph = karate_club_graph()
        self.bow_tie_graph = bow_tie_graph()

    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.louvain.fit(identity(1))

        with self.assertRaises(TypeError):
            self.louvain.fit(identity(2, format='csr'), weights=1)

    def test_unknown_options(self):
        with self.assertRaises(ValueError):
            self.louvain.fit(identity(2, format='csr'), weights='unknown')

    def test_single_node_graph(self):
        self.assertEqual(self.louvain.fit(identity(1, format='csr')).labels_, [0])

    def test_karate_club_graph(self):
        labels = self.louvain.fit(self.karate_club_graph).labels_
        self.assertEqual(labels.shape, (34,))
        self.assertAlmostEqual(modularity(self.karate_club_graph, labels), 0.42, 2)
        if is_numba_available:
            labels = self.louvain_numba.fit(self.karate_club_graph).labels_
            self.assertEqual(labels.shape, (34,))
            self.assertAlmostEqual(modularity(self.karate_club_graph, labels), 0.42, 2)
        labels = self.louvain_high_resolution.fit(self.karate_club_graph).labels_
        self.assertEqual(labels.shape, (34,))
        self.assertAlmostEqual(modularity(self.karate_club_graph, labels), 0.34, 2)
        labels = self.louvain_null_resolution.fit(self.karate_club_graph).labels_
        self.assertEqual(labels.shape, (34,))
        self.assertEqual(self.louvain_null_resolution.n_clusters_, 1)

    def test_shuffling(self):
        self.louvain_shuffle_first.fit(self.bow_tie_graph)
        self.assertEqual(self.louvain_shuffle_first.labels_[1], 0)
        self.louvain_shuffle_second.fit(self.bow_tie_graph)
        self.assertEqual(self.louvain_shuffle_second.labels_[1], 1)
