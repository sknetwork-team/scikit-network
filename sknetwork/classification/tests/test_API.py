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

            classifiers = [PageRankClassifier(), DiffusionClassifier(), DirichletClassifier(),
                           KNN(embedding_method=LouvainEmbedding(), n_neighbors=1), Propagation()]

            with self.assertRaises(ValueError):
                classifiers[0].score(0)

            for algo in classifiers:
                labels1 = algo.fit_transform(adjacency, seeds_array)
                labels2 = algo.fit_transform(adjacency, seeds_dict)
                scores = algo.score(0)
                self.assertTrue((labels1 == labels2).all())
                self.assertEqual(labels2.shape, (n,))
                self.assertTupleEqual(algo.membership_.shape, (n, 2))
                self.assertEqual(scores.shape, (n,))

            seeds1 = {0: 0, 1: 1}
            seeds2 = {0: 0, 1: 2}
            for clf in classifiers:
                labels1 = (clf.fit_transform(adjacency, seeds1) == 1)
                labels2 = (clf.fit_transform(adjacency, seeds2) == 2)
                self.assertTrue((labels1 == labels2).all())
