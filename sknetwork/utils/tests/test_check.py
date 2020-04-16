#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for check.py"""

import unittest

from sknetwork.data import cyclic_digraph
from sknetwork.data.test_graphs import test_graph_disconnect
from sknetwork.utils.check import *


class TestChecks(unittest.TestCase):

    def setUp(self):
        """Simple graphs for tests."""
        self.adjacency = cyclic_digraph(3)
        self.dense_mat = np.identity(3)

    def test_check_format(self):
        with self.assertRaises(TypeError):
            check_format(self.adjacency.tocsc())

    def test_check_square(self):
        with self.assertRaises(ValueError):
            check_square(np.ones((3, 7)))

    def test_check_connected(self):
        with self.assertRaises(ValueError):
            check_connected(test_graph_disconnect())

    def test_non_negative_entries(self):
        self.assertTrue(has_nonnegative_entries(self.adjacency))
        self.assertTrue(has_nonnegative_entries(self.dense_mat))

    def test_positive_entries(self):
        self.assertFalse(has_positive_entries(self.dense_mat))
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            has_positive_entries(self.adjacency)

    def test_probas(self):
        self.assertTrue(is_proba_array(np.array([.5, .5])))
        self.assertEqual(0.5, check_is_proba(0.5))
        with self.assertRaises(TypeError):
            is_proba_array(np.ones((2, 2, 2)))

    def test_error_make_weights(self):
        with self.assertRaises(ValueError):
            make_weights(distribution='junk', adjacency=self.adjacency)

    def test_error_check_is_proba(self):
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            check_is_proba('junk')
        with self.assertRaises(ValueError):
            check_is_proba(2)

    def test_error_check_weights(self):
        with self.assertRaises(ValueError):
            check_weights(np.zeros(4), self.adjacency)
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            check_weights(2, self.adjacency)
        with self.assertRaises(ValueError):
            check_weights(np.zeros(3), self.adjacency, positive_entries=True)
        with self.assertRaises(ValueError):
            check_weights(-np.ones(3), self.adjacency)

    def test_random_state(self):
        random_state = np.random.RandomState(1)
        self.assertEqual(type(check_random_state(random_state)), np.random.RandomState)

    def test_error_random_state(self):
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            check_random_state('junk')

    def test_check_seeds(self):
        n = 10
        seeds_array = -np.ones(n)
        seeds_array[:2] = np.arange(2)
        seeds_dict = {0: 0, 1: 1}
        labels_array = check_seeds(seeds_array, n)
        labels_dict = check_seeds(seeds_dict, n)

        self.assertTrue(np.allclose(labels_array, labels_dict))
        with self.assertRaises(ValueError):
            check_seeds(labels_array, 5)
        with self.assertWarns(Warning):
            seeds_dict[0] = -1
            check_seeds(seeds_dict, n)

    def test_check_labels(self):
        with self.assertRaises(ValueError):
            check_labels(np.ones(3))
        labels = np.ones(5)
        labels[0] = 0
        classes, n_classes = check_labels(labels)
        self.assertTrue(np.equal(classes, np.arange(2)).all())
        self.assertEqual(n_classes, 2)

    def test_check_n_jobs(self):
        self.assertEqual(check_n_jobs(None), 1)
        self.assertEqual(check_n_jobs(-1), None)
        self.assertEqual(check_n_jobs(8), 8)

    def test_check_n_neighbors(self):
        with self.assertWarns(Warning):
            check_n_neighbors(10, 5)

    def test_adj_vector(self):
        n = 10
        vector1 = np.random.rand(n)
        vector2 = sparse.csr_matrix(vector1)
        adj1 = check_adjacency_vector(vector1)
        adj2 = check_adjacency_vector(vector2)

        self.assertAlmostEqual(np.linalg.norm(adj1 - adj2), 0)
        self.assertEqual(adj1.shape, (1, n))

        with self.assertRaises(ValueError):
            check_adjacency_vector(vector1, 2 * n)
