#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
"""

import unittest
import numpy as np
from scipy.sparse import identity
from sknetwork.hierarchy.paris import Paris
from sknetwork.toy_graphs.graph_data import karate_club_graph


class TestParis(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()
        self.karate_club = karate_club_graph()

    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.paris.fit(identity(1))

        with self.assertRaises(TypeError):
            self.paris.fit(identity(2, format='csr'), node_weights=1)

    def test_unknown_options(self):
        with self.assertRaises(ValueError):
            self.paris.fit(identity(2, format='csr'), node_weights='unknown')

    def test_karate_club_graph(self):
        self.paris.fit(self.karate_club)
        self.assertEqual(self.paris.dendrogram_.shape[0], 33)
        self.paris.predict()
        self.assertEqual(np.max(self.paris.labels_), 1)
