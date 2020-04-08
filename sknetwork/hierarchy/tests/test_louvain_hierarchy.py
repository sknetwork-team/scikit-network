#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2020
@author: Quentin Lutz <qlutz@enst.fr>
"""

import unittest

from sknetwork.hierarchy.louvain_hierarchy import LouvainHierarchy
from sknetwork.data.toy_graphs import karate_club


# noinspection PyMissingOrEmptyDocstring
class TestLouvainHierarchy(unittest.TestCase):

    def test_louvain_hierarchy(self):
        lh = LouvainHierarchy()
        karate_graph = karate_club()
        lh.fit(karate_graph)
        self.assertEqual(len(lh.dendrogram_), karate_graph.shape[0] - 1)

