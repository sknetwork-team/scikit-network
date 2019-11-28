# -*- coding: utf-8 -*-
# tests for graph_data.py
"""authors: Quentin Lutz <qlutz@enst.fr>"""

import unittest

import numpy as np

from sknetwork.data.graph_models import block_model


class TestGraphImport(unittest.TestCase):

    def test_errors(self):
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            block_model('foo')

        with self.assertRaises(ValueError):
            block_model(-1)

        with self.assertRaises(ValueError):
            block_model(10)

    # noinspection DuplicatedCode
    def test_generation(self):
        adj, g_t_f, g_t_s = block_model(2, shape=(2, 2), random_state=1)
        self.assertEqual((adj.indptr == [0, 0, 1]).all(), True)
        self.assertEqual((adj.indices == [0]).all(), True)
        self.assertEqual((adj.data == [True]).all(), True)
        self.assertEqual((g_t_f == [0, 1]).all(), True)
        self.assertEqual((g_t_s == [0, 1]).all(), True)

        adj, g_t_f, g_t_s = block_model(np.ones((2, 2), dtype=int), shape=(2, 2), random_state=1)
        self.assertEqual((adj.indptr == [0, 0, 1]).all(), True)
        self.assertEqual((adj.indices == [0]).all(), True)
        self.assertEqual((adj.data == [True]).all(), True)
        self.assertEqual((g_t_f == [0, 1]).all(), True)
        self.assertEqual((g_t_s == [0, 1]).all(), True)
