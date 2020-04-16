#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for spring embeddings"""

import unittest

from sknetwork.embedding import FruchtermanReingold
from sknetwork.data.test_graphs import *


class TestEmbeddings(unittest.TestCase):

    def test_shape(self):
        for adjacency in [test_graph(), test_digraph()]:
            n = adjacency.shape[0]
            spring = FruchtermanReingold()
            layout = spring.fit_transform(adjacency)
            self.assertEqual((n, 2), layout.shape)

    def test_pos_init(self):
        adjacency = test_graph()
        n = adjacency.shape[0]

        spring = FruchtermanReingold(pos_init='anystring')
        layout = spring.fit_transform(adjacency)
        layout = spring.fit_transform(adjacency, pos_init=layout)
        self.assertEqual((n, 2), layout.shape)
