#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for gnn utils"""

import unittest

from sknetwork.gnn.utils import *


class TestUtils(unittest.TestCase):

    def test_check_norm(self):
        with self.assertRaises(ValueError):
            check_normalizations('foo')
        with self.assertRaises(ValueError):
            check_normalizations(['foo', 'bar'])

    def test_mask_similarity(self):
        m1 = np.array([True, True, False])
        m2 = np.array([True, False, False])
        with self.assertWarns(Warning):
            check_mask_similarity(m1, m2)

    def test_early_stopping(self):
        self.assertTrue(check_early_stopping(True, np.array([True, False]), 2))
        self.assertFalse(check_early_stopping(True, None, 2))
        self.assertFalse(check_early_stopping(True, np.array([True, False, True]), None))
        self.assertFalse(check_early_stopping(True, np.array([False, False, False]), 5))

    def test_check_existing_masks(self):
        # Check invalid entries
        labels = np.array([1, 1, 0])
        with self.assertRaises(ValueError):
            check_existing_masks(labels)
        # Check valid size parameters
        self.assertFalse(check_existing_masks(labels, None, None, None, 0.1, 0.2, 0.7)[0])

        # Check valid pre-computed masks
        mask_exists, train_mask, val_mask, test_mask = check_existing_masks(labels, np.array([False, True, False]))
        self.assertTrue(mask_exists)
        self.assertTrue(all(train_mask == np.array([False, True, False])))
        self.assertTrue(all(val_mask == np.array([False, False, False])))
        self.assertTrue(all(test_mask == np.array([True, False, True])))

        mask_exists, train_mask, val_mask, test_mask = check_existing_masks(labels, np.array([True, False, False]),
                                                                            np.array([False, True, False]))
        self.assertTrue(mask_exists)
        self.assertTrue(all(train_mask == np.array([True, False, False])))
        self.assertTrue(all(val_mask == np.array([False, True, False])))
        self.assertTrue(all(test_mask == np.array([False, False, True])))

        mask_exists, train_mask, val_mask, test_mask = check_existing_masks(labels, np.array([True, True, False]),
                                                                            None, np.array([False, False, True]))
        self.assertTrue(mask_exists)
        self.assertTrue(all(train_mask == np.array([True, True, False])))
        self.assertTrue(all(val_mask == np.array([False, False, False])))
        self.assertTrue(all(test_mask == np.array([False, False, True])))

        mask_exists, train_mask, val_mask, test_mask = check_existing_masks(labels, np.array([True, False, False]),
                                                                            np.array([False, False, True]),
                                                                            np.array([False, True, False]))
        self.assertTrue(mask_exists)
        self.assertTrue(all(train_mask == np.array([True, False, False])))
        self.assertTrue(all(val_mask == np.array([False, False, True])))
        self.assertTrue(all(test_mask == np.array([False, True, False])))

        # Check negative labels
        labels = np.array([1, -1, 0])
        mask_exists, train_mask, val_mask, test_mask = check_existing_masks(labels, np.array([True, True, False]))
        self.assertTrue(mask_exists)
        self.assertTrue(all(train_mask == np.array([True, False, False])))
        print(val_mask)
        self.assertTrue(all(val_mask == np.array([False, False, False])))
        self.assertTrue(all(test_mask == np.array([False, True, True])))

    def test_get_layers(self):
        with self.assertRaises(ValueError):
            get_layers([4, 2], 'Conv',  activations=['Relu', 'Sigmoid', 'Relu'], use_bias=True, normalizations='Both',
                       self_embeddings=True, sample_sizes=5, loss=None)
        # Type compatibility
        layers = get_layers([4], 'Conv', activations=['Relu'], use_bias=[True], normalizations=['Both'],
                            self_embeddings=[True], sample_sizes=[5], loss='Cross entropy')
        self.assertTrue(len(np.ravel(layers)) == 1)
        # Broadcasting parameters
        layers = get_layers([4, 2], ['Conv', 'Conv'], activations='Relu', use_bias=True, normalizations='Both',
                            self_embeddings=True, sample_sizes=5, loss='Cross entropy')
        self.assertTrue(len(layers) == 2)

    def test_check_loss(self):
        layer = get_layers([4], 'Conv', activations=['Relu'], use_bias=[True], normalizations=['Both'],
                            self_embeddings=[True], sample_sizes=[5], loss=None)
        with self.assertRaises(ValueError):
            check_loss(layer[0])
