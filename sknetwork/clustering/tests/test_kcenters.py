#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for KCenters"""
import unittest

from sknetwork.clustering import KCenters
from sknetwork.data import karate_club, painters, star_wars
from sknetwork.data.test_graphs import *


class TestKCentersClustering(unittest.TestCase):

    def test_kcenters(self):
        # Test undirected graph
        n_clusters = 2
        adjacency = karate_club()
        n_row = adjacency.shape[0]
        kcenters = KCenters(n_clusters=n_clusters)
        labels = kcenters.fit_predict(adjacency)
        self.assertEqual(len(labels), n_row)
        self.assertEqual(len(set(labels)), n_clusters)

        # Test directed graph
        n_clusters = 3
        adjacency = painters()
        n_row = adjacency.shape[0]
        kcenters = KCenters(n_clusters=n_clusters, directed=True)
        labels = kcenters.fit_predict(adjacency)
        self.assertEqual(len(labels), n_row)
        self.assertEqual(len(set(labels)), n_clusters)

        # Test bipartite graph
        n_clusters = 2
        biadjacency = star_wars()
        n_row, n_col = biadjacency.shape
        kcenters = KCenters(n_clusters=n_clusters)
        kcenters.fit(biadjacency)
        labels = kcenters.labels_
        self.assertEqual(len(kcenters.labels_row_), n_row)
        self.assertEqual(len(kcenters.labels_col_), n_col)
        self.assertEqual(len(set(labels)), n_clusters)

    def test_kcenters_centers(self):
        # Test centers for undirected graphs
        n_clusters = 2
        adjacency = karate_club()
        kcenters = KCenters(n_clusters=n_clusters)
        kcenters.fit(adjacency)
        centers = kcenters.centers_
        self.assertEqual(n_clusters, len(set(centers)))

        # Test centers for bipartite graphs
        n_clusters = 2
        biadjacency = star_wars()
        n_row, n_col = biadjacency.shape
        for position in ["row", "col", "both"]:
            kcenters = KCenters(n_clusters=n_clusters, center_position=position)
            kcenters.fit(biadjacency)
            centers_row = kcenters.centers_row_
            centers_col = kcenters.centers_col_
            if position == "row":
                self.assertEqual(n_clusters, len(set(centers_row)))
                self.assertTrue(np.all(centers_row < n_row))
                self.assertTrue(centers_col is None)
            if position == "col":
                self.assertEqual(n_clusters, len(set(centers_col)))
                self.assertTrue(np.all((centers_col < n_col) & (0 <= centers_col)))
                self.assertTrue(centers_row is None)
            if position == "both":
                self.assertEqual(n_clusters, len(set(centers_row)) + len(set(centers_col)))
                self.assertTrue(np.all(centers_row < n_row))
                self.assertTrue(np.all((centers_col < n_col) & (0 <= centers_col)))

    def test_kcenters_error(self):
        # Test value errors
        adjacency = karate_club()
        biadjacency = star_wars()

        # test n_clusters error
        kcenters = KCenters(n_clusters=1)
        with self.assertRaises(ValueError):
            kcenters.fit(adjacency)

        # test n_init error
        kcenters = KCenters(n_clusters=2, n_init=0)
        with self.assertRaises(ValueError):
            kcenters.fit(adjacency)

        # test center_position error
        kcenters = KCenters(n_clusters=2, center_position="other")
        with self.assertRaises(ValueError):
            kcenters.fit(biadjacency)
