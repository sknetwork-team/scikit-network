#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for parsing.py"""

import unittest
from os import remove

import numpy as np

from sknetwork.data import parsing


class TestTSVParser(unittest.TestCase):

    def test_unlabeled_unweighted(self):
        self.stub_data_1 = 'stub_1.txt'
        with open(self.stub_data_1, "w") as text_file:
            text_file.write('%stub\n1 3\n4 5\n0 2')
        adj = parsing.parse_tsv(self.stub_data_1)
        self.assertEqual(sum(adj.indices == [2, 3, 0, 1, 5, 4]), 6)
        self.assertEqual(sum(adj.indptr == [0, 1, 2, 3, 4, 5, 6]), 7)
        self.assertEqual(sum(adj.data == [1, 1, 1, 1, 1, 1]), 6)
        remove(self.stub_data_1)

    def test_labeled_weighted(self):
        self.stub_data_2 = 'stub_2.txt'
        with open(self.stub_data_2, "w") as text_file:
            text_file.write('%stub\nf, e, 5\na, d, 6\nc, b, 1')
        adj, labels = parsing.parse_tsv(self.stub_data_2)
        self.assertEqual(sum(adj.indices == [4, 3, 5, 1, 0, 2]), 6)
        self.assertEqual(sum(adj.indptr == [0, 1, 2, 3, 4, 5, 6]), 7)
        self.assertEqual(sum(adj.data == [1, 6, 5, 6, 1, 5]), 6)
        self.assertEqual(sum(np.array(list(labels.values())) == [' b', ' d', ' e', 'a', 'c', 'f']), 6)
        remove(self.stub_data_2)

    def test_auto_reindex(self):
        self.stub_data_4 = 'stub_4.txt'
        with open(self.stub_data_4, "w") as text_file:
            text_file.write('%stub\n14 31\n42 50\n0 12')
        adj, labels = parsing.parse_tsv(self.stub_data_4)
        self.assertEqual(sum(adj.indices == [1, 0, 3, 2, 5, 4]), 6)
        self.assertEqual(sum(adj.indptr == [0, 1, 2, 3, 4, 5, 6]), 7)
        self.assertEqual(sum(adj.data == [1, 1, 1, 1, 1, 1]), 6)
        self.assertEqual(sum(np.array(list(labels.values())) == [0, 12, 14, 31, 42, 50]), 6)
        remove(self.stub_data_4)

    def test_wrong_format(self):
        self.stub_data_3 = 'stub_3.txt'
        with open(self.stub_data_3, "w") as text_file:
            text_file.write('%stub\n1 3 a\n4 5 b\n0 2 e')
        self.assertRaises(ValueError, parsing.parse_tsv, self.stub_data_3)
        remove(self.stub_data_3)
