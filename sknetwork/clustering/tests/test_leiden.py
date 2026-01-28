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
        adjacency = test_graph()
        leiden_d = Leiden(modularity='dugue')
        leiden_n = Leiden(modularity='newman')
        labels_d = leiden_d.fit_predict(adjacency)
        labels_n = leiden_n.fit_predict(adjacency)
        self.assertTrue((labels_d == labels_n).all())

    def test_bipartite(self):
        biadjacency = test_bigraph()
        adjacency = bipartite2undirected(biadjacency)
        leiden = Leiden(modularity='newman')
        labels1 = leiden.fit_predict(adjacency)
        leiden.fit(biadjacency)
        labels2 = np.concatenate((leiden.labels_row_, leiden.labels_col_))
        self.assertTrue((labels1 == labels2).all())

    def test_initial_labels(self):
        """Test seeded initialization with array format for Leiden."""
        adjacency = karate_club()
        n_nodes = adjacency.shape[0]

        # Create initial labels with 3 clusters
        initial_labels = np.zeros(n_nodes, dtype=int)
        initial_labels[10:20] = 1
        initial_labels[25:] = 2

        leiden = Leiden(random_state=42)
        labels = leiden.fit_predict(adjacency, initial_labels=initial_labels)

        # Check that we have a valid clustering
        self.assertEqual(len(labels), n_nodes)
        self.assertTrue(len(set(labels)) >= 1)

    def test_initial_labels_validation(self):
        """Test input validation for initial_labels parameter in Leiden."""
        adjacency = karate_club()
        n_nodes = adjacency.shape[0]
        leiden = Leiden()

        # Test wrong length array
        with self.assertRaises(ValueError):
            initial_labels = np.array([0, 1])  # Too short
            leiden.fit_predict(adjacency, initial_labels=initial_labels)

    def test_initial_labels_with_shuffle(self):
        """Test interaction with shuffle_nodes parameter for Leiden."""
        adjacency = karate_club()
        n_nodes = adjacency.shape[0]

        # Create initial labels
        initial_labels = np.zeros(n_nodes, dtype=int)
        initial_labels[10:20] = 1
        initial_labels[25:] = 2

        leiden = Leiden(shuffle_nodes=True, random_state=42)
        labels = leiden.fit_predict(adjacency, initial_labels=initial_labels)

        # Check that we have a valid clustering
        self.assertEqual(len(labels), n_nodes)
        self.assertTrue(len(set(labels)) >= 1)

    def test_initial_labels_reproducibility(self):
        """Test that same initial_labels + random_state produces identical results."""
        adjacency = karate_club()
        n_nodes = adjacency.shape[0]

        initial_labels = np.zeros(n_nodes, dtype=int)
        initial_labels[10:20] = 1
        initial_labels[25:] = 2

        leiden1 = Leiden(random_state=42)
        labels1 = leiden1.fit_predict(adjacency, initial_labels=initial_labels)

        leiden2 = Leiden(random_state=42)
        labels2 = leiden2.fit_predict(adjacency, initial_labels=initial_labels)
        self.assertTrue((labels1 == labels2).all())

    def test_initial_labels_bipartite(self):
        """Test seeded initialization with bipartite graphs for Leiden."""
        biadjacency = star_wars()
        n_row, n_col = biadjacency.shape
        n_nodes = n_row + n_col

        # Create initial labels for the full bipartite graph
        initial_labels = np.zeros(n_nodes, dtype=int)
        initial_labels[n_row//2:n_row] = 1  # Split rows
        initial_labels[n_row + n_col//2:] = 2  # Split cols

        leiden = Leiden(random_state=42)
        leiden.fit(biadjacency, initial_labels=initial_labels)

        # Check that we have valid clustering for both row and col labels
        self.assertTrue(hasattr(leiden, 'labels_row_'))
        self.assertTrue(hasattr(leiden, 'labels_col_'))
        self.assertEqual(len(leiden.labels_row_), n_row)
        self.assertEqual(len(leiden.labels_col_), n_col)

    def test_initial_labels_bipartite_row_col(self):
        """Test seeded initialization with separate row/col parameters for bipartite graphs (Leiden)."""
        biadjacency = star_wars()
        n_row, n_col = biadjacency.shape

        # Create row and col labels separately
        initial_labels_row = np.zeros(n_row, dtype=int)
        initial_labels_row[n_row//2:] = 1

        initial_labels_col = np.zeros(n_col, dtype=int)
        initial_labels_col[n_col//2:] = 2

        leiden = Leiden(random_state=42)
        leiden.fit(biadjacency, initial_labels_row=initial_labels_row, initial_labels_col=initial_labels_col)

        # Check that we have valid clustering for both row and col labels
        self.assertTrue(hasattr(leiden, 'labels_row_'))
        self.assertTrue(hasattr(leiden, 'labels_col_'))
        self.assertEqual(len(leiden.labels_row_), n_row)
        self.assertEqual(len(leiden.labels_col_), n_col)

    def test_initial_labels_row_col_validation(self):
        """Test validation for row/col parameters (Leiden)."""
        adjacency = test_graph()
        biadjacency = star_wars()
        n_row, n_col = biadjacency.shape
        leiden = Leiden()

        # Test: cannot use both initial_labels and initial_labels_row/col together
        with self.assertRaises(ValueError):
            initial_labels = np.zeros(n_row + n_col)
            initial_labels_row = np.zeros(n_row)
            leiden.fit(biadjacency, initial_labels=initial_labels, initial_labels_row=initial_labels_row)

        # Test: cannot use row/col parameters with non-bipartite graphs
        with self.assertRaises(ValueError):
            initial_labels_row = np.zeros(10)
            leiden.fit(adjacency, initial_labels_row=initial_labels_row)
