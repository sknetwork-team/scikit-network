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

    def test_initial_labels_array(self):
        """Test seeded initialization with array format."""
        adjacency = karate_club()
        n_nodes = adjacency.shape[0]

        # Create initial labels with 3 clusters
        initial_labels = np.zeros(n_nodes, dtype=int)
        initial_labels[10:20] = 1
        initial_labels[25:] = 2

        louvain = Louvain(random_state=42)
        labels = louvain.fit_predict(adjacency, initial_labels=initial_labels)

        # Check that we have a valid clustering
        self.assertEqual(len(labels), n_nodes)
        self.assertTrue(len(set(labels)) >= 1)

    def test_initial_labels_dict(self):
        """Test seeded initialization with dictionary format."""
        adjacency = karate_club()
        n_nodes = adjacency.shape[0]

        # Create initial labels with dict - sparse assignment
        initial_labels = {0: 0, 10: 1, 20: 2, 30: 3}

        louvain = Louvain(random_state=42)
        labels = louvain.fit_predict(adjacency, initial_labels=initial_labels)

        # Check that we have a valid clustering
        self.assertEqual(len(labels), n_nodes)
        self.assertTrue(len(set(labels)) >= 1)

    def test_initial_labels_validation(self):
        """Test input validation for initial_labels parameter."""
        adjacency = karate_club()
        n_nodes = adjacency.shape[0]
        louvain = Louvain()

        # Test wrong length array
        with self.assertRaises(ValueError):
            initial_labels = np.array([0, 1])  # Too short
            louvain.fit_predict(adjacency, initial_labels=initial_labels)

        # Test negative labels
        with self.assertRaises(ValueError):
            initial_labels = np.array([-1] * n_nodes)
            louvain.fit_predict(adjacency, initial_labels=initial_labels)

        # Test invalid node index in dict
        with self.assertRaises(ValueError):
            initial_labels = {n_nodes: 0}  # Index out of range
            louvain.fit_predict(adjacency, initial_labels=initial_labels)

    def test_initial_labels_backward_compatibility(self):
        """Ensure existing behavior is preserved when initial_labels=None."""
        adjacency = karate_club()

        # Test with default (None) should behave exactly as before
        louvain1 = Louvain(random_state=42)
        labels1 = louvain1.fit_predict(adjacency)

        louvain2 = Louvain(random_state=42)
        labels2 = louvain2.fit_predict(adjacency, initial_labels=None)

        # Should produce identical results
        np.testing.assert_array_equal(labels1, labels2)

    def test_initial_labels_with_shuffle(self):
        """Test interaction with shuffle_nodes parameter."""
        adjacency = karate_club()
        n_nodes = adjacency.shape[0]

        # Create initial labels
        initial_labels = np.zeros(n_nodes, dtype=int)
        initial_labels[10:20] = 1
        initial_labels[25:] = 2

        louvain = Louvain(shuffle_nodes=True, random_state=42)
        labels = louvain.fit_predict(adjacency, initial_labels=initial_labels)

        # Check that we have a valid clustering
        self.assertEqual(len(labels), n_nodes)
        self.assertTrue(len(set(labels)) >= 1)

    def test_initial_labels_bipartite(self):
        """Test seeded initialization with bipartite graphs."""
        biadjacency = star_wars()
        n_row, n_col = biadjacency.shape
        n_nodes = n_row + n_col

        # Create initial labels for the full bipartite graph
        initial_labels = np.zeros(n_nodes, dtype=int)
        initial_labels[n_row//2:n_row] = 1  # Split rows
        initial_labels[n_row + n_col//2:] = 2  # Split cols

        louvain = Louvain(random_state=42)
        louvain.fit(biadjacency, initial_labels=initial_labels)

        # Check that we have valid clustering for both row and col labels
        self.assertTrue(hasattr(louvain, 'labels_row_'))
        self.assertTrue(hasattr(louvain, 'labels_col_'))
        self.assertEqual(len(louvain.labels_row_), n_row)
        self.assertEqual(len(louvain.labels_col_), n_col)

    def test_initial_labels_label_reindexing(self):
        """Test that labels are properly reindexed to start from 0."""
        adjacency = karate_club()
        n_nodes = adjacency.shape[0]

        # Create initial labels with gaps (non-consecutive)
        initial_labels = np.full(n_nodes, 10, dtype=int)  # All nodes in cluster 10
        initial_labels[10:20] = 100  # Some nodes in cluster 100
        initial_labels[25:] = 500   # Some nodes in cluster 500

        louvain = Louvain(random_state=42)
        labels = louvain.fit_predict(adjacency, initial_labels=initial_labels)

        # Check that final labels start from 0 and are consecutive
        unique_labels = sorted(set(labels))
        expected_labels = list(range(len(unique_labels)))
        self.assertEqual(unique_labels, expected_labels)

    def test_initial_labels_reproducibility(self):
        """Test that same initial_labels + random_state produces identical results."""
        adjacency = karate_club()
        n_nodes = adjacency.shape[0]

        initial_labels = np.zeros(n_nodes, dtype=int)
        initial_labels[10:20] = 1
        initial_labels[25:] = 2

        louvain1 = Louvain(random_state=42)
        labels1 = louvain1.fit_predict(adjacency, initial_labels=initial_labels)

        louvain2 = Louvain(random_state=42)
        labels2 = louvain2.fit_predict(adjacency, initial_labels=initial_labels)

        # Should produce identical results
        np.testing.assert_array_equal(labels1, labels2)

    def test_initial_labels_bipartite_row_col(self):
        """Test seeded initialization with separate row/col parameters for bipartite graphs."""
        biadjacency = star_wars()
        n_row, n_col = biadjacency.shape

        # Create row and col labels separately
        initial_labels_row = np.zeros(n_row, dtype=int)
        initial_labels_row[n_row//2:] = 1

        initial_labels_col = np.zeros(n_col, dtype=int)
        initial_labels_col[n_col//2:] = 2

        louvain = Louvain(random_state=42)
        louvain.fit(biadjacency, initial_labels_row=initial_labels_row, initial_labels_col=initial_labels_col)

        # Check that we have valid clustering for both row and col labels
        self.assertTrue(hasattr(louvain, 'labels_row_'))
        self.assertTrue(hasattr(louvain, 'labels_col_'))
        self.assertEqual(len(louvain.labels_row_), n_row)
        self.assertEqual(len(louvain.labels_col_), n_col)

    def test_initial_labels_bipartite_row_col_dict(self):
        """Test seeded initialization with dictionary format for row/col parameters."""
        biadjacency = star_wars()
        n_row, n_col = biadjacency.shape

        # Create row and col labels as dictionaries
        initial_labels_row = {0: 0, n_row//2: 1}
        initial_labels_col = {0: 2, n_col//2: 3}

        louvain = Louvain(random_state=42)
        louvain.fit(biadjacency, initial_labels_row=initial_labels_row, initial_labels_col=initial_labels_col)

        # Check that we have valid clustering for both row and col labels
        self.assertTrue(hasattr(louvain, 'labels_row_'))
        self.assertTrue(hasattr(louvain, 'labels_col_'))
        self.assertEqual(len(louvain.labels_row_), n_row)
        self.assertEqual(len(louvain.labels_col_), n_col)

    def test_initial_labels_bipartite_row_only(self):
        """Test seeded initialization with only row labels for bipartite graphs."""
        biadjacency = star_wars()
        n_row, n_col = biadjacency.shape

        # Only specify row labels, col labels should default to identity
        initial_labels_row = np.zeros(n_row, dtype=int)
        initial_labels_row[n_row//2:] = 1

        louvain = Louvain(random_state=42)
        louvain.fit(biadjacency, initial_labels_row=initial_labels_row)

        self.assertTrue(hasattr(louvain, 'labels_row_'))
        self.assertTrue(hasattr(louvain, 'labels_col_'))
        self.assertEqual(len(louvain.labels_row_), n_row)
        self.assertEqual(len(louvain.labels_col_), n_col)

    def test_initial_labels_bipartite_col_only(self):
        """Test seeded initialization with only col labels for bipartite graphs."""
        biadjacency = star_wars()
        n_row, n_col = biadjacency.shape

        # Only specify col labels, row labels should default to identity
        initial_labels_col = np.zeros(n_col, dtype=int)
        initial_labels_col[n_col//2:] = 1

        louvain = Louvain(random_state=42)
        louvain.fit(biadjacency, initial_labels_col=initial_labels_col)

        self.assertTrue(hasattr(louvain, 'labels_row_'))
        self.assertTrue(hasattr(louvain, 'labels_col_'))
        self.assertEqual(len(louvain.labels_row_), n_row)
        self.assertEqual(len(louvain.labels_col_), n_col)

    def test_initial_labels_row_col_validation(self):
        """Test validation for row/col parameters."""
        adjacency = karate_club()
        biadjacency = star_wars()
        n_row, n_col = biadjacency.shape
        louvain = Louvain()

        # Test: cannot use both initial_labels and initial_labels_row/col together
        with self.assertRaises(ValueError):
            initial_labels = np.zeros(n_row + n_col)
            initial_labels_row = np.zeros(n_row)
            louvain.fit(biadjacency, initial_labels=initial_labels, initial_labels_row=initial_labels_row)

        # Test: cannot use row/col parameters with non-bipartite graphs
        with self.assertRaises(ValueError):
            initial_labels_row = np.zeros(10)
            louvain.fit(adjacency, initial_labels_row=initial_labels_row)

        # Test: wrong length for row labels
        with self.assertRaises(ValueError):
            initial_labels_row = np.array([0, 1])  # Too short
            louvain.fit(biadjacency, initial_labels_row=initial_labels_row)

        # Test: wrong length for col labels
        with self.assertRaises(ValueError):
            initial_labels_col = np.array([0])  # Too short
            louvain.fit(biadjacency, initial_labels_col=initial_labels_col)

        # Test: negative labels in row
        with self.assertRaises(ValueError):
            initial_labels_row = np.array([-1] * n_row)
            louvain.fit(biadjacency, initial_labels_row=initial_labels_row)

        # Test: negative labels in col
        with self.assertRaises(ValueError):
            initial_labels_col = np.array([-1] * n_col)
            louvain.fit(biadjacency, initial_labels_col=initial_labels_col)

        # Test: invalid node index in row dict
        with self.assertRaises(ValueError):
            initial_labels_row = {n_row: 0}  # Index out of range
            louvain.fit(biadjacency, initial_labels_row=initial_labels_row)

        # Test: invalid node index in col dict
        with self.assertRaises(ValueError):
            initial_labels_col = {n_col: 0}  # Index out of range
            louvain.fit(biadjacency, initial_labels_col=initial_labels_col)
