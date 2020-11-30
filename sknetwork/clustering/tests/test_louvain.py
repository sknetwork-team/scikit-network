#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Louvain"""
import unittest

from sknetwork.clustering import Louvain, BiLouvain
from sknetwork.data import karate_club, star_wars
from sknetwork.data.test_graphs import *
from sknetwork.utils import bipartite2undirected


class TestLouvainClustering(unittest.TestCase):

    def test_disconnected(self):
        adjacency = test_graph_disconnect()
        n = adjacency.shape[0]
        labels = Louvain().fit_transform(adjacency)
        self.assertEqual(len(labels), n)

    def test_modularity(self):
        adjacency = karate_club()
        louvain_d = Louvain(modularity='dugue')
        louvain_n = Louvain(modularity='newman')
        labels_d = louvain_d.fit_transform(adjacency)
        labels_n = louvain_n.fit_transform(adjacency)
        self.assertTrue((labels_d == labels_n).all())

        louvain_p = Louvain(modularity='potts')
        louvain_p.fit_transform(adjacency)

    def test_bilouvain(self):
        biadjacency = star_wars()
        adjacency = bipartite2undirected(biadjacency)

        louvain = Louvain(modularity='newman')
        bilouvain = BiLouvain(modularity='newman')

        labels1 = louvain.fit_transform(adjacency)
        bilouvain.fit(biadjacency)
        labels2 = np.concatenate((bilouvain.labels_row_, bilouvain.labels_col_))
        self.assertTrue((labels1 == labels2).all())

    def test_options(self):
        adjacency = karate_club()

        # resolution
        louvain = Louvain(resolution=2)
        labels = louvain.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # tolerance
        louvain = Louvain(resolution=2, tol_aggregation=0.1)
        labels = louvain.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 12)

        # shuffling
        louvain = Louvain(resolution=2, shuffle_nodes=True, random_state=42)
        labels = louvain.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 9)

        # aggregate graph
        louvain = Louvain(return_aggregate=True)
        labels = louvain.fit_transform(adjacency)
        n_labels = len(set(labels))
        self.assertEqual(louvain.adjacency_.shape, (n_labels, n_labels))

        # aggregate graph
        Louvain(n_aggregations=1, sort_clusters=False).fit(adjacency)

    def test_options_with_64_bit(self):
        adjacency = karate_club()
        # force 64-bit index
        adjacency.indices = adjacency.indices.astype(np.int64)
        adjacency.indptr = adjacency.indptr.astype(np.int64)

        # resolution
        louvain = Louvain(resolution=2)
        labels = louvain.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 7)

        # tolerance
        louvain = Louvain(resolution=2, tol_aggregation=0.1)
        labels = louvain.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 12)

        # shuffling
        louvain = Louvain(resolution=2, shuffle_nodes=True, random_state=42)
        labels = louvain.fit_transform(adjacency)
        self.assertEqual(len(set(labels)), 9)

        # aggregate graph
        louvain = Louvain(return_aggregate=True)
        labels = louvain.fit_transform(adjacency)
        n_labels = len(set(labels))
        self.assertEqual(louvain.adjacency_.shape, (n_labels, n_labels))

        # aggregate graph
        Louvain(n_aggregations=1, sort_clusters=False).fit(adjacency)

        # check if labels are 64-bit
        self.assertEqual(labels.dtype, np.int64)

    def test_invalid(self):
        adjacency = karate_club()
        louvain = Louvain(modularity='toto')
        with self.assertRaises(ValueError):
            louvain.fit(adjacency)
