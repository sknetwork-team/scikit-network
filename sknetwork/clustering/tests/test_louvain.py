#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Louvain"""

import unittest

from scipy import sparse

from sknetwork import is_numba_available
from sknetwork.clustering import Louvain, BiLouvain, modularity
from sknetwork.toy_graphs import random_graph, karate_club, bow_tie, painters, star_wars_villains
from sknetwork.utils.adjacency_formats import directed2undirected


class TestLouvainClustering(unittest.TestCase):

    def setUp(self):
        self.louvain = Louvain()
        self.bilouvain = BiLouvain()
        self.louvain_high_resolution = Louvain(engine='python', resolution=2)
        self.louvain_null_resolution = Louvain(engine='python', resolution=0)
        if is_numba_available:
            self.louvain_numba = Louvain(engine='numba')
            self.bilouvain_numba = BiLouvain(engine='numba')
        else:
            with self.assertRaises(ValueError):
                Louvain(engine='numba')
        self.louvain_shuffle_first = Louvain(engine='python', shuffle_nodes=True, random_state=0)
        self.louvain_shuffle_second = Louvain(engine='python', shuffle_nodes=True, random_state=123)
        self.random_graph = random_graph()
        self.karate_club: sparse.csr_matrix = karate_club()
        self.bow_tie = bow_tie()
        self.painters = painters(return_labels=False)
        self.star_wars: sparse.csr_matrix = star_wars_villains()

    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.louvain.fit(sparse.identity(1))

    def test_single_node_graph(self):
        self.louvain.fit(sparse.identity(1, format='csr'))
        self.assertEqual(self.louvain.labels_, [0])

    def test_random_graph(self):
        self.louvain.fit(directed2undirected(self.random_graph))
        self.assertEqual(len(self.louvain.labels_), 10)

    def test_undirected(self):
        self.louvain.fit(self.karate_club)
        labels = self.louvain.labels_
        self.assertEqual(labels.shape, (34,))
        self.assertAlmostEqual(modularity(self.karate_club, labels), 0.42, 2)
        if is_numba_available:
            self.louvain_numba.fit(self.karate_club)
            labels = self.louvain_numba.labels_
            self.assertEqual(labels.shape, (34,))
            self.assertAlmostEqual(modularity(self.karate_club, labels), 0.42, 2)
        self.louvain_high_resolution.fit(self.karate_club)
        labels = self.louvain_high_resolution.labels_
        self.assertEqual(labels.shape, (34,))
        self.assertAlmostEqual(modularity(self.karate_club, labels), 0.34, 2)
        self.louvain_null_resolution.fit(self.karate_club)
        labels = self.louvain_null_resolution.labels_
        self.assertEqual(labels.shape, (34,))
        self.assertEqual(len(set(self.louvain_null_resolution.labels_)), 1)

    def test_directed(self):
        self.louvain.fit(self.painters)
        labels = self.louvain.labels_
        self.assertEqual(labels.shape, (14,))
        self.assertAlmostEqual(modularity(self.painters, labels), 0.32, 2)

        self.bilouvain.fit(self.painters)
        n1, n2 = self.painters.shape
        labels = self.bilouvain.labels_
        col_labels = self.bilouvain.col_labels_
        self.assertEqual(labels.shape, (n1,))
        self.assertEqual(col_labels.shape, (n2,))

    def test_bipartite(self):
        self.bilouvain.fit(self.star_wars)
        labels = self.bilouvain.labels_
        feature_labels = self.bilouvain.col_labels_
        self.assertEqual(labels.shape, (4,))
        self.assertEqual(feature_labels.shape, (3,))
        if is_numba_available:
            self.bilouvain_numba.fit(self.star_wars)
            labels = self.bilouvain_numba.labels_
            self.assertEqual(labels.shape, (4,))

    def test_shuffling(self):
        self.louvain_shuffle_first.fit(self.bow_tie)
        self.assertEqual(self.louvain_shuffle_first.labels_[1], 1)
        self.louvain_shuffle_second.fit(self.bow_tie)
        self.assertEqual(self.louvain_shuffle_second.labels_[1], 1)
