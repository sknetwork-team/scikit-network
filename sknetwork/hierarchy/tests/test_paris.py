#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import unittest
import numpy as np
from scipy.sparse import identity
from sknetwork.hierarchy import Paris
from sknetwork.toy_graphs import house_graph, karate_club_graph


class TestParis(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()
        self.house_graph = house_graph()
        self.karate_club_graph = karate_club_graph()

    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.paris.fit(identity(1))

        with self.assertRaises(TypeError):
            self.paris.fit(identity(2, format='csr'), node_weights=1)

    def test_unknown_options(self):
        with self.assertRaises(ValueError):
            self.paris.fit(identity(2, format='csr'), node_weights='unknown')

    def test_house_graph(self):
        self.paris.fit(self.house_graph)
        self.assertEqual(self.paris.dendrogram_.shape[0], 4)
        labels = self.paris.predict(sorted_clusters=True)
        self.assertTrue(np.array_equal(labels, np.array([0, 0, 1, 1, 0])))

    def test_karate_club_graph(self):
        self.paris.fit(self.karate_club_graph)
        self.assertEqual(self.paris.dendrogram_.shape[0], 33)
        labels = self.paris.predict()
        self.assertEqual(np.max(labels), 1)
