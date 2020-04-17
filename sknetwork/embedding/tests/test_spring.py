#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for spring embeddings"""

import unittest

from sknetwork.embedding import Spring
from sknetwork.data.test_graphs import *


class TestEmbeddings(unittest.TestCase):

    def test_shape(self):
        for adjacency in [test_graph(), test_digraph()]:
            n = adjacency.shape[0]
            spring = Spring()
            layout = spring.fit_transform(adjacency)
            self.assertEqual((n, 2), layout.shape)

    def test_pos_init(self):
        adjacency = test_graph()
        n = adjacency.shape[0]

        spring = Spring(position_init='random')
        layout = spring.fit_transform(adjacency)
        self.assertEqual((n, 2), layout.shape)
        layout = spring.fit_transform(adjacency, position_init=layout)
        self.assertEqual((n, 2), layout.shape)
