#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for LouvainNE"""
import unittest

from sknetwork.data.test_graphs import test_bigraph, test_graph
from sknetwork.embedding import BiLouvainNE, LouvainNE


class TestLouvainEmbedding(unittest.TestCase):

    def test_louvainne(self):
        lne = LouvainNE()
        lne.fit(test_graph())
        self.assertTupleEqual(lne.embedding_.shape, (10, 2))

    def test_bilouvainne(self):
        blne = BiLouvainNE()
        blne.fit(test_bigraph())
        self.assertTupleEqual(blne.embedding_.shape, (6, 2))
