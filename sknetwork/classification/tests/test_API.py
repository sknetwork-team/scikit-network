#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for classification API"""

import unittest

from sknetwork.classification import *
from sknetwork.data.test_graphs import *
from sknetwork.embedding import LouvainEmbedding


class TestClassificationAPI(unittest.TestCase):

    def test_undirected(self):
        for adjacency in [test_graph(), test_digraph()]:
            n = adjacency.shape[0]
            seeds_array = -np.ones(n)
            seeds_array[:2] = np.arange(2)
            seeds_dict = {0: 0, 1: 1}

            classifiers = [PageRankClassifier(), DiffusionClassifier(),
                           NNClassifier(embedding_method=LouvainEmbedding(), n_neighbors=1), Propagation()]

            for algo in classifiers:
                labels1 = algo.fit_predict(adjacency, seeds_array)
                labels2 = algo.fit_predict(adjacency, seeds_dict)
                self.assertTrue((labels1 == labels2).all())
                self.assertEqual(labels2.shape, (n,))
                membership = algo.fit_transform(adjacency, seeds_array)
                self.assertTupleEqual(membership.shape, (n, 2))
