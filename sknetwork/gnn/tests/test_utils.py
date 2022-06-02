#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for gnn utils"""

import unittest

import numpy as np

from sknetwork.gnn.utils import check_existing_masks, check_norm


class TestUtils(unittest.TestCase):

    def test_check_norm(self):
        with self.assertRaises(ValueError):
            check_norm('toto')

    def test_existing_masks(self):
        # Check invalid entries
        with self.assertRaises(ValueError):
            check_existing_masks(None, np.array([True, False, True]), None)
        with self.assertRaises(ValueError):
            check_existing_masks(None, None, None)
        with self.assertRaises(ValueError):
            check_existing_masks(None, None, -1)
        # Check valid test_size parameter
        self.assertFalse(check_existing_masks(None, None, 0.2))
        # Check valid pre-computed masks
        self.assertTrue(check_existing_masks(np.array([True, False, True]), np.array([False, True, False]), 0.2))
        self.assertTrue(check_existing_masks(np.array([True, False, True]), np.array([False, True, False]), -1))
        self.assertTrue(check_existing_masks(np.array([True, False, True]), np.array([False, True, False]), None))
