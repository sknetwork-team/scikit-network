#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for bilouvain.py"""


import unittest
from sknetwork.clustering import *
from sknetwork.toy_graphs import *


class TestBiLouvainClustering(unittest.TestCase):

    def setUp(self):
        self.bilouvain = BiLouvain()
        self.bilouvain_high_resolution = BiLouvain(resolution=2)
        self.star_wars_graph = star_wars_villains_graph()

    def test_star_wars_graph(self):
        labels = self.bilouvain.fit(self.star_wars_graph).sample_labels_
        self.assertEqual(labels.shape, (4,))
        labels = self.bilouvain_high_resolution.fit(self.star_wars_graph).sample_labels_
        self.assertEqual(labels.shape, (4,))
