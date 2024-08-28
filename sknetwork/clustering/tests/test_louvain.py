#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Louvain"""
import unittest

from sknetwork.clustering import Louvain
from sknetwork.data import karate_club, star_wars
from sknetwork.data.test_graphs import *
from sknetwork.utils import bipartite2undirected


class TestLouvainClustering(unittest.TestCase):

    def test_disconnected(self):
        adjacency = test_disconnected_graph()
        n = adjacency.shape[0]
        labels = Louvain().fit_predict(adjacency)
        self.assertEqual(len(labels), n)

    def test_format(self):
        adjacency = test_graph()
        n = adjacency.shape[0]
        labels = Louvain().fit_predict(adjacency.toarray())
        self.assertEqual(len(labels), n)

    def test_modularity(self):
        adjacency = karate_club()
        louvain_d = Louvain(modularity='dugue')
        louvain_n = Louvain(modularity='newman')
        labels_d = louvain_d.fit_predict(adjacency)
        labels_n = louvain_n.fit_predict(adjacency)
        self.assertTrue((labels_d == labels_n).all())
        louvain_p = Louvain(modularity='potts')
        louvain_p.fit_predict(adjacency)

    def test_bilouvain(self):
        biadjacency = star_wars()
        adjacency = bipartite2undirected(biadjacency)
        louvain = Louvain(modularity='newman')
        labels1 = louvain.fit_predict(adjacency)
        louvain.fit(biadjacency)
        labels2 = np.concatenate((louvain.labels_row_, louvain.labels_col_))
        self.assertTrue((labels1 == labels2).all())

    def test_options(self):
        adjacency = karate_club()

        # resolution
        louvain = Louvain(resolution=2)
        labels = louvain.fit_predict(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # tolerance
        louvain = Louvain(resolution=2, tol_aggregation=0.1)
        labels = louvain.fit_predict(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # shuffling
        louvain = Louvain(resolution=2, shuffle_nodes=True, random_state=42)
        labels = louvain.fit_predict(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # aggregate graph
        louvain = Louvain(return_aggregate=True)
        labels = louvain.fit_predict(adjacency)
        n_labels = len(set(labels))
        self.assertEqual(louvain.aggregate_.shape, (n_labels, n_labels))

        # aggregate graph
        Louvain(n_aggregations=1, sort_clusters=False).fit(adjacency)

    def test_options_with_64_bit(self):
        adjacency = karate_club()
        # force 64-bit index
        adjacency.indices = adjacency.indices.astype(np.int64)
        adjacency.indptr = adjacency.indptr.astype(np.int64)

        # resolution
        louvain = Louvain(resolution=2)
        labels = louvain.fit_predict(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # tolerance
        louvain = Louvain(resolution=2, tol_aggregation=0.1)
        labels = louvain.fit_predict(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # shuffling
        louvain = Louvain(resolution=2, shuffle_nodes=True, random_state=42)
        labels = louvain.fit_predict(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # aggregate graph
        louvain = Louvain(return_aggregate=True)
        labels = louvain.fit_predict(adjacency)
        n_labels = len(set(labels))
        self.assertEqual(louvain.aggregate_.shape, (n_labels, n_labels))

        # aggregate graph
        Louvain(n_aggregations=1, sort_clusters=False).fit(adjacency)

        # check if labels are 64-bit
        self.assertEqual(labels.dtype, np.int64)

    def test_predict(self):
        adjacency = karate_club()
        n_nodes = adjacency.shape[0]
        louvain = Louvain()
        labels = louvain.fit_predict(adjacency)
        self.assertEqual(len(labels), n_nodes)
        probs = louvain.fit_predict_proba(adjacency)
        self.assertEqual(probs.shape[0], n_nodes)
        membership = louvain.fit_transform(adjacency)
        self.assertEqual(membership.shape[0], n_nodes)
        biadjacency = star_wars()
        n_row, n_col = biadjacency.shape
        louvain.fit(biadjacency)
        labels = louvain.predict()
        self.assertEqual(len(labels), n_row)
        labels = louvain.predict(columns=True)
        self.assertEqual(len(labels), n_col)
        probs = louvain.predict_proba()
        self.assertEqual(probs.shape[0], n_row)
        probs = louvain.predict(columns=True)
        self.assertEqual(probs.shape[0], n_col)
        membership = louvain.transform()
        self.assertEqual(membership.shape[0], n_row)
        membership = louvain.transform(columns=True)
        self.assertEqual(membership.shape[0], n_col)

    def test_invalid(self):
        adjacency = karate_club()
        louvain = Louvain(modularity='toto')
        with self.assertRaises(ValueError):
            louvain.fit(adjacency)
