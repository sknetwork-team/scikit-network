#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for loss"""

import unittest

from sknetwork.gnn.loss import *


class TestLoss(unittest.TestCase):

    def test_get_loss_function(self):
        self.assertTrue(isinstance(get_loss_function('CrossEntropyLoss'), type(cross_entropy_loss)))
        with self.assertRaises(ValueError):
            get_loss_function('toto')

    def test_get_loss_function_prime(self):
        self.assertTrue(isinstance(get_prime_loss_function('CrossEntropyLoss'), type(cross_entropy_prime)))
        with self.assertRaises(ValueError):
            get_prime_loss_function('toto')

    def test_ce_loss(self):
        # Results compared to sklearn.metrics.log_loss for ndim = 1
        logits = np.array([0.1, 0.15, 0.2, 0.35, 0.2])
        y_true = np.array([0, 0, 0, 0, 1])
        self.assertAlmostEqual(cross_entropy_loss(y_true, logits), 0.506248764999)

        # ndim > 1
        logits = np.array([[0.2, 0.2, 0.6], [0., 0.3, 0.7], [0.4, 0.3, 0.3]])
        y_true = np.array([0, 2, 1])
        self.assertAlmostEqual(cross_entropy_loss(y_true, logits), 1.0566952202329)

    def test_ce_prime(self):
        # ndim > 1
        logits = np.array([[0.2, 0.2, 0.6], [0., 0.3, 0.7], [0.4, 0.3, 0.3]])
        y_true_ohe = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        self.assertAlmostEqual((cross_entropy_prime(y_true_ohe, logits) - (logits - y_true_ohe)).sum(), 0)
