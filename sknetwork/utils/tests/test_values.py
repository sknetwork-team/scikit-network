#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for values.py"""

import unittest

import numpy as np

from sknetwork.utils.values import get_values, stack_values, values2prob


class TestValues(unittest.TestCase):

    def test_get_values(self):
        n = 10
        labels_array = -np.ones(n)
        labels_array[:2] = np.arange(2)
        labels_dict = {0: 0, 1: 1}
        labels_array = get_values((n,), labels_array)
        labels_ = get_values((n,), labels_dict)
        self.assertTrue(np.allclose(labels_array, labels_))
        with self.assertRaises(ValueError):
            get_values((5,), labels_array)
        self.assertRaises(TypeError, get_values, 'toto', 3)
        with self.assertWarns(Warning):
            labels_dict[0] = -1
            get_values((n,), labels_dict)

    def test_values2probs(self):
        n = 4
        values_array = np.array([0, 1, -1, 0])
        values_dict = {0: 0, 1: 1, 3: 0}

        probs1 = values2prob(n, values_array)
        probs2 = values2prob(n, values_dict)
        self.assertTrue(np.allclose(probs1, probs2))

        bad_input = np.array([0, 0, -1, 0])
        with self.assertRaises(ValueError):
            values2prob(n, bad_input)

    def test_stack_values(self):
        shape = 4, 3
        values_row_array = np.array([0, 1, -1, 0])
        values_row_dict = {0: 0, 1: 1, 3: 0}
        values_col_array = np.array([0, 1, -1])
        values_col_dict = {0: 0, 1: 1}

        values1 = stack_values(shape, values_row_array, values_col_array)
        values2 = stack_values(shape, values_row_dict, values_col_dict)
        values3 = stack_values(shape, values_row_array, values_col_dict)
        values4 = stack_values(shape, values_row_dict, values_col_array)

        self.assertTrue(np.allclose(values1, values2))
        self.assertTrue(np.allclose(values2, values3))
        self.assertTrue(np.allclose(values3, values4))

        values1 = stack_values(shape, values_row_array, None)
        values2 = stack_values(shape, values_row_dict, None)

        self.assertTrue(np.allclose(values1, values2))

        values1 = stack_values(shape, None, values_col_array)
        values2 = stack_values(shape, None, values_col_dict)

        self.assertTrue(np.allclose(values1, values2))
