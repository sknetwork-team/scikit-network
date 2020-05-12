#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2020
@author: Quentin Lutz <qlutz@enst.fr>
@author: Thomas Bonald <tbonald@enst.fr>
"""

import unittest

from sknetwork.hierarchy.louvain_hierarchy import LouvainHierarchy
from sknetwork.data.test_graphs import test_graph


class TestLouvainHierarchy(unittest.TestCase):

    def test_options(self):
        louvain = LouvainHierarchy(resolution=2)
        adjacency = test_graph()
        dendrogram = louvain.fit_transform(adjacency)
        n = adjacency.shape[0]
        self.assertEqual(dendrogram.shape, (n - 1, 4))

        louvain = LouvainHierarchy(depth=1)
        adjacency = test_graph()
        dendrogram = louvain.fit_transform(adjacency)
        n = adjacency.shape[0]
        self.assertEqual(dendrogram.shape, (n - 1, 4))
        self.assertEqual(dendrogram[:, 2].max(), 1)

