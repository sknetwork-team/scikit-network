#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for gnn utils"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.gnn.utils import check_existing_masks, check_normalizations, check_layers, get_layers, \
    has_self_loops, add_self_loops


class TestUtils(unittest.TestCase):

    def test_check_norm(self):
        with self.assertRaises(ValueError):
            check_normalizations('toto')
        with self.assertRaises(ValueError):
            check_normalizations(['toto', 'toto'])

    def test_check_existing_masks(self):
        # Check invalid entries
        labels = np.array([1, 1, 0])
        with self.assertRaises(ValueError):
            check_existing_masks(labels, None, None)
        # Check valid test_size parameter
        self.assertFalse(check_existing_masks(labels, None, 0.2)[0])
        # Check valid pre-computed masks
        mask_exists, train_mask, val_mask, test_mask = check_existing_masks(labels, np.array([False, True, False]), 0.2)
        self.assertTrue(mask_exists)
        self.assertTrue(all(train_mask == np.array([False, True, False])))
        self.assertTrue(all(val_mask == np.array([True, False, True])))
        self.assertTrue(all(test_mask == np.array([False, False, False])))
        mask_exists, train_mask, val_mask, test_mask = check_existing_masks(labels, np.array([False, True, False]))
        self.assertTrue(mask_exists)
        self.assertTrue(all(train_mask == np.array([False, True, False])))
        self.assertTrue(all(val_mask == np.array([True, False, True])))
        self.assertTrue(all(test_mask == np.array([False, False, False])))
        # Check negative labels
        labels = np.array([1, -1, 0])
        mask_exists, train_mask, val_mask, test_mask = check_existing_masks(labels, np.array([True, True, False]))
        self.assertTrue(mask_exists)
        self.assertTrue(all(train_mask == np.array([True, False, False])))
        self.assertTrue(all(val_mask == np.array([False, False, True])))
        self.assertTrue(all(test_mask == np.array([False, True, False])))

    def test_check_layer(self):
        with self.assertRaises(ValueError):
            check_layers('toto')
        with self.assertRaises(ValueError):
            check_layers(['toto', 'toto'])

    def test_check_layers_parameters(self):
        with self.assertRaises(ValueError):
            get_layers([4, 2], 'GCNConv',  activations=['Relu', 'Sigmoid', 'Relu'], use_bias=True,
                       normalizations='Both', self_loops=True)
        # Type compatibility
        params = get_layers([4], 'GCNConv', activations=['Relu'],
                            use_bias=[True], normalizations=['Both'], self_loops=[True])
        self.assertTrue(len(np.ravel(params)) == 6)
        # Broadcasting parameters
        params = get_layers([4, 2], ['GCNConv', 'GCNConv'], activations='Relu',
                            use_bias=True, normalizations='Both', self_loops=True)
        self.assertTrue(len(np.ravel(params)) == 12)

    def test_has_self_loops(self):
        self.assertTrue(has_self_loops(sparse.csr_matrix(np.array([[1, 0], [1, 1]]))))
        self.assertFalse(has_self_loops(sparse.csr_matrix(np.array([[0, 0], [1, 1]]))))

    def test_add_self_loops(self):
        # Square adjacency
        adjacency = sparse.csr_matrix(np.array([[0, 0], [1, 1]]))
        self.assertFalse(has_self_loops(adjacency))
        adjacency = add_self_loops(adjacency)
        self.assertTrue(has_self_loops(adjacency))
        # Non square adjacency
        adjacency = sparse.csr_matrix(np.array([[0, 0, 1], [1, 1, 1]]))
        n_row, n_col = adjacency.shape
        self.assertTrue(has_self_loops(add_self_loops(adjacency)[:, :n_row]))
