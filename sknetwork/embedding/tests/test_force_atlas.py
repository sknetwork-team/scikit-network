#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for force atlas2 embeddings"""
import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph, test_digraph
from sknetwork.embedding.force_atlas import ForceAtlas


class TestEmbeddings(unittest.TestCase):

    def test_options(self):
        for adjacency in [test_graph(), test_digraph()]:
            n = adjacency.shape[0]

            force_atlas = ForceAtlas()
            layout = force_atlas.fit_transform(adjacency)
            self.assertEqual((n, 2), layout.shape)

            force_atlas = ForceAtlas(lin_log=True)
            layout = force_atlas.fit_transform(adjacency)
            self.assertEqual((n, 2), layout.shape)

            force_atlas = ForceAtlas(approx_radius=1.)
            layout = force_atlas.fit_transform(adjacency)
            self.assertEqual((n, 2), layout.shape)

            force_atlas.fit(adjacency, pos_init=layout, n_iter=1)

    def test_errors(self):
        adjacency = test_graph()
        with self.assertRaises(ValueError):
            ForceAtlas().fit(adjacency, pos_init=np.ones((5, 7)))
