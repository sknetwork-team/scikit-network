#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for seeds.py"""

import unittest

import numpy as np

from sknetwork.utils.seeds import get_seeds, stack_seeds, seeds2probs


class TestSeeds(unittest.TestCase):

    def test_get_seeds(self):
        n = 10
        seeds_array = -np.ones(n)
        seeds_array[:2] = np.arange(2)
        seeds_dict = {0: 0, 1: 1}
        labels_array = get_seeds((n,), seeds_array)
        labels_dict = get_seeds((n,), seeds_dict)

        self.assertTrue(np.allclose(labels_array, labels_dict))
        with self.assertRaises(ValueError):
            get_seeds((5,), labels_array)
        self.assertRaises(TypeError, get_seeds, 'toto', 3)
        with self.assertWarns(Warning):
            seeds_dict[0] = -1
            get_seeds((n,), seeds_dict)

    def test_seeds2probs(self):
        n = 4
        seeds_array = np.array([0, 1, -1, 0])
        seeds_dict = {0: 0, 1: 1, 3: 0}

        probs1 = seeds2probs(n, seeds_array)
        probs2 = seeds2probs(n, seeds_dict)
        self.assertTrue(np.allclose(probs1, probs2))

        bad_input = np.array([0, 0, -1, 0])
        with self.assertRaises(ValueError):
            seeds2probs(n, bad_input)

    def test_stack_seeds(self):
        shape = 4, 3
        seeds_row_array = np.array([0, 1, -1, 0])
        seeds_row_dict = {0: 0, 1: 1, 3: 0}
        seeds_col_array = np.array([0, 1, -1])
        seeds_col_dict = {0: 0, 1: 1}

        seeds1 = stack_seeds(shape, seeds_row_array, seeds_col_array)
        seeds2 = stack_seeds(shape, seeds_row_dict, seeds_col_dict)
        seeds3 = stack_seeds(shape, seeds_row_array, seeds_col_dict)
        seeds4 = stack_seeds(shape, seeds_row_dict, seeds_col_array)

        self.assertTrue(np.allclose(seeds1, seeds2))
        self.assertTrue(np.allclose(seeds2, seeds3))
        self.assertTrue(np.allclose(seeds3, seeds4))

        seeds1 = stack_seeds(shape, seeds_row_array, None)
        seeds2 = stack_seeds(shape, seeds_row_dict, None)

        self.assertTrue(np.allclose(seeds1, seeds2))

        seeds1 = stack_seeds(shape, None, seeds_col_array)
        seeds2 = stack_seeds(shape, None, seeds_col_dict)

        self.assertTrue(np.allclose(seeds1, seeds2))
