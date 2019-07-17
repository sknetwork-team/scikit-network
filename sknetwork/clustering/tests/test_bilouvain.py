#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for bilouvain.py"""


import unittest
from sknetwork.clustering import BiLouvain
from sknetwork.toy_graphs import star_wars_villains
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
        self.bilouvain_null_resolution = BiLouvain(resolution=0., verbose=False)

        self.star_wars_graph = star_wars_villains()

    def test_star_wars_graph(self):
        self.bilouvain.fit(self.star_wars_graph)
        labels = self.bilouvain.labels_
        self.assertEqual(labels.shape, (4,))
        self.bilouvain_high_resolution.fit(self.star_wars_graph)
        labels = self.bilouvain_high_resolution.labels_
        self.assertEqual(labels.shape, (4,))
        if is_numba_available:
            self.bilouvain_numba.fit(self.star_wars_graph)
            labels = self.bilouvain_numba.labels_
            self.assertEqual(labels.shape, (4,))
        self.bilouvain_null_resolution.fit(self.star_wars_graph)
        self.assertEqual(len(set(self.bilouvain_null_resolution.labels_)), 1)
