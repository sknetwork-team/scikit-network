# -*- coding: utf-8 -*-
# test for paris.py

"""test for Paris algorithm"""

import unittest
from sknetwork.hierarchy.paris import Paris
from sknetwork.toy_graphs.graph_data import karate_club_graph
from scipy.sparse import identity


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
        self.assertEqual(self.paris.fit(self.karate_club).dendrogram_.shape[0], 33)
