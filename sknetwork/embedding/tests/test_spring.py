#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for spring embeddings"""

import unittest

from sknetwork.data.test_graphs import *
from sknetwork.embedding import Spring


class TestEmbeddings(unittest.TestCase):

    def test_shape(self):
        for adjacency in [test_graph(), test_digraph()]:
            n = adjacency.shape[0]
            spring = Spring()
            layout = spring.fit_transform(adjacency)
            self.assertEqual((n, 2), layout.shape)

            spring = Spring(n_components=3)
            layout = spring.fit_transform(adjacency)
            self.assertEqual((n, 3), layout.shape)

    def test_pos_init(self):
        adjacency = test_graph()
        n = adjacency.shape[0]

        spring = Spring(strength=0.1, position_init='spectral', tol=1e3)
        layout = spring.fit_transform(adjacency)
        self.assertEqual((n, 2), layout.shape)
        layout = spring.fit_transform(adjacency, position_init=layout)
        self.assertEqual((n, 2), layout.shape)

    def test_approx_radius(self):
        adjacency = test_graph()
        n = adjacency.shape[0]

        spring = Spring(approx_radius=1.)
        layout = spring.fit_transform(adjacency)
        self.assertEqual((n, 2), layout.shape)

    def test_errors(self):
        adjacency = test_graph()
        with self.assertRaises(ValueError):
            Spring(position_init='toto')
        with self.assertRaises(ValueError):
            Spring().fit(adjacency, position_init=np.ones((2, 2)))
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            Spring().fit(adjacency, position_init='toto')
