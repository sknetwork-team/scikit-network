#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork import is_numba_available
from sknetwork.hierarchy import Paris, straight_cut
from sknetwork.toy_graphs import house, karate_club


# noinspection PyMissingOrEmptyDocstring
class TestParis(unittest.TestCase):

    def setUp(self):
        self.paris_python = Paris(engine='python')

    # noinspection PyTypeChecker
    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.paris_python.fit(sparse.identity(1))

        with self.assertRaises(TypeError):
            self.paris_python.fit(sparse.identity(2, format='csr'), custom_weights=1)

    # noinspection PyTypeChecker
    def test_unknown_options(self):
        with self.assertRaises(ValueError):
            self.paris_python.fit(sparse.identity(2, format='csr'), custom_weights='unknown')

    # noinspection DuplicatedCode
    def test_house_graph(self):
        self.house_graph = house()
        if is_numba_available:
            self.paris_numba = Paris(engine='numba')
        else:
            with self.assertRaises(ValueError):
                Paris(engine='numba')
        if is_numba_available:
            self.paris_numba.fit(self.house_graph)
            self.assertEqual(self.paris_numba.dendrogram_.shape[0], 4)
            labels = straight_cut(self.paris_numba.dendrogram_, sorted_clusters=True)
            self.assertTrue(np.array_equal(labels, np.array([0, 0, 1, 1, 0])))
        self.paris_python.fit(self.house_graph)
        self.assertEqual(self.paris_python.dendrogram_.shape[0], 4)
        labels = straight_cut(self.paris_python.dendrogram_, sorted_clusters=True)
        self.assertTrue(np.array_equal(labels, np.array([0, 0, 1, 1, 0])))

    def test_karate_club_graph(self):
        self.karate_club_graph: sparse.csr_matrix = karate_club()
        self.paris_python.fit(self.karate_club_graph)
        self.assertEqual(self.paris_python.dendrogram_.shape[0], 33)
        labels = straight_cut(self.paris_python.dendrogram_)
        self.assertEqual(np.max(labels), 1)
