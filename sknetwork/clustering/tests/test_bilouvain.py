#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for bilouvain.py"""


import unittest
from sknetwork.clustering import *
from sknetwork.toy_graphs import *
from sknetwork import is_numba_available


class TestBiLouvainClustering(unittest.TestCase):

    def setUp(self):
        self.bilouvain = BiLouvain()
        self.bilouvain_high_resolution = BiLouvain(resolution=2, engine='python')
        if is_numba_available:
            self.bilouvain_numba = BiLouvain(engine='numba')
        else:
            with self.assertRaises(ValueError):
                BiLouvain(engine='numba')
        self.biouvain_null_resolution = BiLouvain(resolution=0., verbose=False)

        self.star_wars_graph = star_wars_villains_graph()

    def test_star_wars_graph(self):
        labels = self.bilouvain.fit(self.star_wars_graph).labels_
        self.assertEqual(labels.shape, (4,))
        labels = self.bilouvain_high_resolution.fit(self.star_wars_graph).labels_
        self.assertEqual(labels.shape, (4,))
        if is_numba_available:
            labels = self.bilouvain_numba.fit(self.star_wars_graph).labels_
            self.assertEqual(labels.shape, (4,))
        self.biouvain_null_resolution.fit(self.star_wars_graph)
        self.assertEqual(self.biouvain_null_resolution.n_clusters_, 1)
