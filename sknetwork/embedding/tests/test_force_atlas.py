#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for force atlas2 embeddings"""

import unittest

from sknetwork.embedding import force_atlas
from sknetwork.data.test_graphs import *


class TestEmbeddings(unittest.TestCase):

    def test_shape(self):
        for adjacency in [test_graph(), test_digraph()]:
            n = adjacency.shape[0]
            force_atlas_graph = force_atlas.ForceAtlas2()
            layout = force_atlas_graph.fit_transform(adjacency)
            self.assertEqual((n, 2), layout.shape)
