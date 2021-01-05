#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for LouvainNE"""
import unittest

from sknetwork.data.test_graphs import test_graph
from sknetwork.embedding import LouvainNE


class TestLouvainEmbedding(unittest.TestCase):

    def test_louvainne(self):
        lne = LouvainNE()
        lne.fit(test_graph())
        self.assertTupleEqual(lne.embedding_.shape, (10, 2))
