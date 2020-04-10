#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""

import unittest

from sknetwork.hierarchy import Ward, cut_straight, cut_balanced, aggregate_dendrogram
from sknetwork.data import karate_club


# noinspection PyMissingOrEmptyDocstring
class TestCuts(unittest.TestCase):

    def setUp(self):
        ward = Ward()
        adjacency = karate_club()
        self.dendrogram = ward.fit_transform(adjacency)

    def test_cuts(self):
        labels = cut_straight(self.dendrogram)
        self.assertEqual(len(set(labels)), 2)
        labels = cut_straight(self.dendrogram, n_clusters=5)
        self.assertEqual(len(set(labels)), 5)
        labels = cut_balanced(self.dendrogram)
        self.assertEqual(len(set(labels)), 22)

    def test_options(self):
        labels = cut_straight(self.dendrogram, sort_clusters=False)
        self.assertEqual(len(set(labels)), 2)
        labels = cut_balanced(self.dendrogram, sort_clusters=False)
        self.assertEqual(len(set(labels)), 22)
        labels = cut_balanced(self.dendrogram, max_cluster_size=10)
        self.assertEqual(len(set(labels)), 6)
        labels, dendrogram = cut_straight(self.dendrogram, n_clusters=7, return_dendrogram=True)
        self.assertEqual(dendrogram.shape, (6, 4))

    def test_aggregation(self):
        aggregated = aggregate_dendrogram(self.dendrogram, n_clusters=3)
        self.assertEqual(len(aggregated), 2)

        aggregated, counts = aggregate_dendrogram(self.dendrogram, n_clusters=3, return_counts=True)
        self.assertEqual(len(aggregated), 2)
        self.assertEqual(len(counts), 3)

