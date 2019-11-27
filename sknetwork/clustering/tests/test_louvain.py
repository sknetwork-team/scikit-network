#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Louvain"""

import unittest

from scipy import sparse

from sknetwork import is_numba_available
from sknetwork.clustering import Louvain, BiLouvain, modularity
from sknetwork.data import simple_directed_graph, karate_club, bow_tie, painters, star_wars_villains
from sknetwork.utils.adjacency_formats import directed2undirected


# noinspection PyMissingOrEmptyDocstring
class TestLouvainClustering(unittest.TestCase):

    def setUp(self):
        self.louvain = Louvain(engine='python')
        self.bilouvain = BiLouvain(engine='python')
        if is_numba_available:
            self.louvain_numba = Louvain(engine='numba')
            self.bilouvain_numba = BiLouvain(engine='numba')
        else:
            with self.assertRaises(ValueError):
                Louvain(engine='numba')

    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.louvain.fit(sparse.identity(1))

    def test_single_node_graph(self):
        self.assertEqual(self.louvain.fit_transform(sparse.identity(1, format='csr')), [0])

    def test_simple_graph(self):
        self.simple_directed_graph = simple_directed_graph()
        self.louvain.fit(directed2undirected(self.simple_directed_graph))
        self.assertEqual(len(self.louvain.labels_), 10)

    def test_undirected(self):
        self.louvain_high_resolution = Louvain(engine='python', resolution=2)
        self.louvain_null_resolution = Louvain(engine='python', resolution=0)
        self.karate_club = karate_club()
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
        self.painters = painters(return_labels=False)

        self.louvain.fit(self.painters)
        labels = self.louvain.labels_
        self.assertEqual(labels.shape, (14,))
        self.assertAlmostEqual(modularity(self.painters, labels), 0.32, 2)

        self.bilouvain.fit(self.painters)
        n1, n2 = self.painters.shape
        row_labels = self.bilouvain.row_labels_
        col_labels = self.bilouvain.col_labels_
        self.assertEqual(row_labels.shape, (n1,))
        self.assertEqual(col_labels.shape, (n2,))

    def test_bipartite(self):
        star_wars_graph = star_wars_villains()
        self.bilouvain.fit(star_wars_graph)
        row_labels = self.bilouvain.row_labels_
        col_labels = self.bilouvain.col_labels_
        self.assertEqual(row_labels.shape, (4,))
        self.assertEqual(col_labels.shape, (3,))
        if is_numba_available:
            self.bilouvain_numba.fit(star_wars_graph)
            row_labels = self.bilouvain_numba.row_labels_
            col_labels = self.bilouvain_numba.col_labels_
            self.assertEqual(row_labels.shape, (4,))
            self.assertEqual(col_labels.shape, (3,))

    def test_shuffling(self):
        self.louvain_shuffle_first = Louvain(engine='python', shuffle_nodes=True, random_state=0)
        self.louvain_shuffle_second = Louvain(engine='python', shuffle_nodes=True, random_state=123)
        self.bow_tie = bow_tie()
        self.louvain_shuffle_first.fit(self.bow_tie)
        self.assertEqual(self.louvain_shuffle_first.labels_[1], 1)
        self.louvain_shuffle_second.fit(self.bow_tie)
        self.assertEqual(self.louvain_shuffle_second.labels_[1], 1)
