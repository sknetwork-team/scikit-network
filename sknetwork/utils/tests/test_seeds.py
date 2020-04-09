#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for seeds.py"""

import unittest

import numpy as np

from sknetwork.utils.seeds import stack_seeds, seeds2probs


class TestSeeds(unittest.TestCase):

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
        n_row, n_col = 4, 3
        seeds_row_array = np.array([0, 1, -1, 0])
        seeds_row_dict = {0: 0, 1: 1, 3: 0}
        seeds_col_array = np.array([0, 1, -1])
        seeds_col_dict = {0: 0, 1: 1}

        seeds1 = stack_seeds(n_row, n_col, seeds_row_array, seeds_col_array)
        seeds2 = stack_seeds(n_row, n_col, seeds_row_dict, seeds_col_dict)
        seeds3 = stack_seeds(n_row, n_col, seeds_row_array, seeds_col_dict)
        seeds4 = stack_seeds(n_row, n_col, seeds_row_dict, seeds_col_array)

        self.assertTrue(np.allclose(seeds1, seeds2))
        self.assertTrue(np.allclose(seeds2, seeds3))
        self.assertTrue(np.allclose(seeds3, seeds4))

        seeds1 = stack_seeds(n_row, n_col, seeds_row_array, None)
        seeds2 = stack_seeds(n_row, n_col, seeds_row_dict, None)

        self.assertTrue(np.allclose(seeds1, seeds2))

        seeds1 = stack_seeds(n_col, n_row, None, seeds_row_array)
        seeds2 = stack_seeds(n_col, n_row, None, seeds_row_dict)

        self.assertTrue(np.allclose(seeds1, seeds2))
