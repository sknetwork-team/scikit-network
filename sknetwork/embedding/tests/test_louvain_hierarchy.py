#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for LouvainNE"""
import unittest

from sknetwork.data.test_graphs import test_graph, test_graph_disconnect, test_digraph, test_bigraph
from sknetwork.embedding import LouvainNE


class TestLouvainNE(unittest.TestCase):

    def test_louvain_hierarchy(self):
        louvain = LouvainNE()
        for adjacency in [test_graph(), test_graph_disconnect(), test_digraph()]:
            self.assertTupleEqual(louvain.fit_transform(adjacency).shape, (10, 2))
        louvain.fit(test_bigraph())
        self.assertTupleEqual(louvain.embedding_.shape, (6, 2))
        louvain.fit(test_graph(), force_bipartite=True)
        self.assertTupleEqual(louvain.embedding_.shape, (10, 2))
