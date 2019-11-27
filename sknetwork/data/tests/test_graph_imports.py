# -*- coding: utf-8 -*-
# tests for graph_data.py
"""authors: Quentin Lutz <qlutz@enst.fr>, Nathan de Lara <ndelara@enst.fr>"""

import unittest

from sknetwork.data.graph_data import karate_club, star_wars_villains, \
    movie_actor, painters, miserables


class TestGraphImport(unittest.TestCase):

    def test_available(self):
        adjacency = karate_club()
        self.assertEqual(adjacency.shape[0], 34)

        biadjacency, _, _ = star_wars_villains(return_labels=True)
        self.assertEqual(biadjacency.shape, (4, 3))

        biadjacency, _, _ = movie_actor(return_labels=True)
        self.assertEqual(biadjacency.shape, (15, 16))

        adjacency, _ = painters(return_labels=True)
        self.assertEqual(adjacency.shape[0], 14)

        adjacency, _ = miserables(return_labels=True)
        self.assertEqual(adjacency.shape[0], 77)
