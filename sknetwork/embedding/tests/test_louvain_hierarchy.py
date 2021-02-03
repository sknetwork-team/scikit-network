#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for LouvainNE"""
import unittest

from sknetwork.data.test_graphs import test_bigraph, test_graph
from sknetwork.embedding import BiHLouvainEmbedding, HLouvainEmbedding


class TestHLouvainEmbedding(unittest.TestCase):

    def test_louvain_hierarchy(self):
        lne = HLouvainEmbedding()
        lne.fit(test_graph())
        self.assertTupleEqual(lne.embedding_.shape, (10, 2))

    def test_bilouvain_hierarchy(self):
        blne = BiHLouvainEmbedding()
        blne.fit(test_bigraph())
        self.assertTupleEqual(blne.embedding_.shape, (6, 2))
