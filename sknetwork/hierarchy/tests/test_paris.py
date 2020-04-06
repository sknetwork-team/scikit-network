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

from sknetwork.hierarchy import Paris, BiParis, straight_cut, balanced_cut
from sknetwork.data import house, karate_club, star_wars_villains


# noinspection PyMissingOrEmptyDocstring
class TestParis(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()
        self.biparis = BiParis()

    # noinspection PyTypeChecker
    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.paris.fit(sparse.identity(1))

    # noinspection DuplicatedCode
    def test_undirected(self):
        house_graph = house()
        self.paris.fit(house_graph)
        self.assertEqual(self.paris.dendrogram_.shape[0], 4)
        labels = straight_cut(self.paris.dendrogram_, sorted_clusters=True)
        self.assertTrue(np.array_equal(labels, np.array([0, 0, 1, 1, 0])))

        karate_club_graph = karate_club()
        self.paris.fit(karate_club_graph)
        self.assertEqual(self.paris.dendrogram_.shape[0], 33)
        labels = straight_cut(self.paris.dendrogram_)
        self.assertEqual(np.max(labels), 1)
        labels = balanced_cut(self.paris.dendrogram_, max_cluster_size=10)
        self.assertEqual(len(set(labels)), 5)

    def test_bipartite(self):
        star_wars_graph = star_wars_villains()
        self.biparis.fit(star_wars_graph)
        dendrogram = self.biparis.dendrogram_
        self.assertEqual(dendrogram.shape, (6, 4))
