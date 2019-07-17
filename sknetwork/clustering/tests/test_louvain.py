#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Louvain"""


import unittest
from sknetwork.clustering import Louvain, modularity
from sknetwork.toy_graphs import karate_club, bow_tie, painters
from scipy.sparse import identity
from sknetwork import is_numba_available


class TestLouvainClustering(unittest.TestCase):

    def setUp(self):
        self.louvain = Louvain()
        self.louvain_high_resolution = Louvain(engine='python', resolution=2)
        self.louvain_null_resolution = Louvain(engine='python', resolution=0)
        if is_numba_available:
            self.louvain_numba = Louvain(engine='numba')
        else:
            with self.assertRaises(ValueError):
                Louvain(engine='numba')
        self.louvain_shuffle_first = Louvain(engine='python', shuffle_nodes=True, random_state=0)
        self.louvain_shuffle_second = Louvain(engine='python', shuffle_nodes=True, random_state=123)
        self.karate_club_graph = karate_club()
        self.bow_tie_graph = bow_tie()
        self.painters_graph = painters(return_labels=False)

    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.louvain.fit(identity(1))

        with self.assertRaises(TypeError):
            self.louvain.fit(identity(2, format='csr'), weights=1)

    def test_unknown_options(self):
        with self.assertRaises(ValueError):
            self.louvain.fit(identity(2, format='csr'), weights='unknown')

    def test_single_node_graph(self):
        self.louvain.fit(identity(1, format='csr'))
        self.assertEqual(self.louvain.labels_, [0])

    def test_undirected(self):
        self.louvain.fit(self.karate_club_graph)
        labels = self.louvain.labels_
        self.assertEqual(labels.shape, (34,))
        self.assertAlmostEqual(modularity(self.karate_club_graph, labels), 0.42, 2)
        if is_numba_available:
            self.louvain_numba.fit(self.karate_club_graph)
            labels = self.louvain_numba.labels_
            self.assertEqual(labels.shape, (34,))
            self.assertAlmostEqual(modularity(self.karate_club_graph, labels), 0.42, 2)
        self.louvain_high_resolution.fit(self.karate_club_graph)
        labels = self.louvain_high_resolution.labels_
        self.assertEqual(labels.shape, (34,))
        self.assertAlmostEqual(modularity(self.karate_club_graph, labels), 0.34, 2)
        self.louvain_null_resolution.fit(self.karate_club_graph)
        labels = self.louvain_null_resolution.labels_
        self.assertEqual(labels.shape, (34,))
        self.assertEqual(len(set(self.louvain_null_resolution.labels_)), 1)

    def test_directed(self):
        self.louvain.fit(self.painters_graph)
        labels = self.louvain.labels_
        self.assertEqual(labels.shape, (14,))
        self.assertAlmostEqual(modularity(self.painters_graph, labels), 0.32, 2)

    def test_shuffling(self):
        self.louvain_shuffle_first.fit(self.bow_tie_graph)
        self.assertEqual(self.louvain_shuffle_first.labels_[1], 0)
        self.louvain_shuffle_second.fit(self.bow_tie_graph)
        self.assertEqual(self.louvain_shuffle_second.labels_[1], 1)
