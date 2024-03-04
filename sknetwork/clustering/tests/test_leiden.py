#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Leiden"""
import unittest

from sknetwork.clustering import Leiden
from sknetwork.data import karate_club, star_wars
from sknetwork.data.test_graphs import *
from sknetwork.utils import bipartite2undirected


class TestLeidenClustering(unittest.TestCase):

    def test_disconnected(self):
        adjacency = test_disconnected_graph()
        n = adjacency.shape[0]
        labels = Leiden().fit_predict(adjacency)
        self.assertEqual(len(labels), n)

    def test_modularity(self):
        adjacency = karate_club()
        leiden_d = Leiden(modularity='dugue')
        leiden_n = Leiden(modularity='newman')
        labels_d = leiden_d.fit_predict(adjacency)
        labels_n = leiden_n.fit_predict(adjacency)
        self.assertTrue((labels_d == labels_n).all())

        leiden_p = Leiden(modularity='potts')
        leiden_p.fit_predict(adjacency)

    def test_bileiden(self):
        biadjacency = star_wars()
        adjacency = bipartite2undirected(biadjacency)
        leiden = Leiden(modularity='newman')
        labels1 = leiden.fit_predict(adjacency)
        leiden.fit(biadjacency)
        labels2 = np.concatenate((leiden.labels_row_, leiden.labels_col_))
        self.assertTrue((labels1 == labels2).all())

    def test_options(self):
        adjacency = karate_club()

        # resolution
        leiden = Leiden(resolution=2)
        labels = leiden.fit_predict(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # tolerance
        leiden = Leiden(resolution=2, tol_aggregation=0.1)
        labels = leiden.fit_predict(adjacency)
        self.assertEqual(len(set(labels)), 12)

        # shuffling
        leiden = Leiden(resolution=2, shuffle_nodes=True, random_state=42)
        labels = leiden.fit_predict(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # aggregate graph
        leiden = Leiden(return_aggregate=True)
        labels = leiden.fit_predict(adjacency)
        n_labels = len(set(labels))
        self.assertEqual(leiden.aggregate_.shape, (n_labels, n_labels))

        # aggregate graph
        Leiden(n_aggregations=1, sort_clusters=False).fit(adjacency)

    def test_options_with_64_bit(self):
        adjacency = karate_club()
        # force 64-bit index
        adjacency.indices = adjacency.indices.astype(np.int64)
        adjacency.indptr = adjacency.indptr.astype(np.int64)

        # resolution
        leiden = Leiden(resolution=2)
        labels = leiden.fit_predict(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # tolerance
        leiden = Leiden(resolution=2, tol_aggregation=0.1)
        labels = leiden.fit_predict(adjacency)
        self.assertEqual(len(set(labels)), 12)

        # shuffling
        leiden = Leiden(resolution=2, shuffle_nodes=True, random_state=42)
        labels = leiden.fit_predict(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # aggregate graph
        leiden = Leiden(return_aggregate=True)
        labels = leiden.fit_predict(adjacency)
        n_labels = len(set(labels))
        self.assertEqual(leiden.aggregate_.shape, (n_labels, n_labels))

        # aggregate graph
        Leiden(n_aggregations=1, sort_clusters=False).fit(adjacency)

        # check if labels are 64-bit
        self.assertEqual(labels.dtype, np.int64)

    def test_predict(self):
        adjacency = karate_club()
        n_nodes = adjacency.shape[0]
        leiden = Leiden()
        labels = leiden.fit_predict(adjacency)
        self.assertEqual(len(labels), n_nodes)
        probs = leiden.fit_predict_proba(adjacency)
        self.assertEqual(probs.shape[0], n_nodes)
        membership = leiden.fit_transform(adjacency)
        self.assertEqual(membership.shape[0], n_nodes)
        biadjacency = star_wars()
        n_row, n_col = biadjacency.shape
        leiden.fit(biadjacency)
        labels = leiden.predict()
        self.assertEqual(len(labels), n_row)
        labels = leiden.predict(columns=True)
        self.assertEqual(len(labels), n_col)
        probs = leiden.predict_proba()
        self.assertEqual(probs.shape[0], n_row)
        probs = leiden.predict(columns=True)
        self.assertEqual(probs.shape[0], n_col)
        membership = leiden.transform()
        self.assertEqual(membership.shape[0], n_row)
        membership = leiden.transform(columns=True)
        self.assertEqual(membership.shape[0], n_col)

    def test_invalid(self):
        adjacency = karate_club()
        leiden = Leiden(modularity='toto')
        with self.assertRaises(ValueError):
            leiden.fit(adjacency)
