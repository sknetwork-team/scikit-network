#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""

import unittest

import numpy as np

from sknetwork.hierarchy import Paris, cut_straight, cut_balanced
from sknetwork.data import karate_club


# noinspection PyMissingOrEmptyDocstring
class TestCuts(unittest.TestCase):

    def setUp(self):
        paris = Paris(engine='python')
        adjacency = karate_club()
        self.dendrogram = paris.fit_transform(adjacency)

    def test_cuts(self):
        labels = cut_straight(self.dendrogram)
        self.assertEqual(len(set(labels)), 2)
        labels = cut_straight(self.dendrogram, n_clusters=5)
        self.assertEqual(len(set(labels)), 5)
        labels = cut_balanced(self.dendrogram)
        self.assertEqual(len(set(labels)), 21)

    def test_options(self):
        labels = cut_straight(self.dendrogram, sort_clusters=False)
        self.assertEqual(len(set(labels)), 2)
        labels = cut_balanced(self.dendrogram, sort_clusters=False)
        self.assertEqual(len(set(labels)), 21)
        labels = cut_balanced(self.dendrogram, max_cluster_size=10)
        self.assertEqual(len(set(labels)), 5)
        labels, dendrogram = cut_straight(self.dendrogram, n_clusters=7, return_dendrogram=True)
        self.assertEqual(dendrogram.shape, (6, 4))

