#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for KNN"""
import unittest

from sknetwork.classification import KNN
from sknetwork.data.test_graphs import *
from sknetwork.embedding import LouvainEmbedding


class TestDiffusionClassifier(unittest.TestCase):

    def test_parallel(self):
        for adjacency in [test_graph(), test_digraph(), test_bigraph()]:
            seeds = {0: 0, 1: 1}

            algo1 = KNN(n_neighbors=1, n_jobs=None, embedding_method=LouvainEmbedding())
            algo2 = KNN(n_neighbors=1, n_jobs=-1, embedding_method=LouvainEmbedding())

            labels1 = algo1.fit_transform(adjacency, seeds)
            labels2 = algo2.fit_transform(adjacency, seeds)

            self.assertTrue(np.allclose(labels1, labels2))

