#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""

import unittest
import numpy as np
from scipy.sparse import identity
from sknetwork.hierarchy import Paris, cut
from sknetwork.toy_graphs import house_graph, karate_club_graph
from sknetwork import is_numba_available


class TestParis(unittest.TestCase):

    def setUp(self):
        self.paris_python = Paris(engine='python')
        self.house_graph = house_graph()
        self.karate_club_graph = karate_club_graph()
        if is_numba_available:
            self.paris_numba = Paris(engine='numba')
        else:
            with self.assertRaises(ValueError):
                Paris(engine='numba')

    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.paris_python.fit(identity(1))

        with self.assertRaises(TypeError):
            self.paris_python.fit(identity(2, format='csr'), weights=1)

    def test_unknown_options(self):
        with self.assertRaises(ValueError):
            self.paris_python.fit(identity(2, format='csr'), weights='unknown')

    def test_house_graph(self):
        if is_numba_available:
            self.paris_numba.fit(self.house_graph)
            self.assertEqual(self.paris_numba.dendrogram_.shape[0], 4)
            labels = cut(self.paris_numba.dendrogram_, sorted_clusters=True)
            self.assertTrue(np.array_equal(labels, np.array([0, 0, 1, 1, 0])))
        self.paris_python.fit(self.house_graph)
        self.assertEqual(self.paris_python.dendrogram_.shape[0], 4)
        labels = cut(self.paris_python.dendrogram_, sorted_clusters=True)
        self.assertTrue(np.array_equal(labels, np.array([0, 0, 1, 1, 0])))

    def test_karate_club_graph(self):
        self.paris_python.fit(self.karate_club_graph)
        self.assertEqual(self.paris_python.dendrogram_.shape[0], 33)
        labels = cut(self.paris_python.dendrogram_)
        self.assertEqual(np.max(labels), 1)
