#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for parser.py"""


import unittest
from sknetwork.loader import parser
import numpy as np
from os import remove


class TestTSVParser(unittest.TestCase):

    def setUp(self):
        self.stub_data_1 = 'stub_1.txt'
        text_file = open(self.stub_data_1, "w")
        text_file.write('%stub\n1 3\n4 5\n0 2')
        text_file.close()

        self.stub_data_2 = 'stub_2.txt'
        text_file = open(self.stub_data_2, "w")
        text_file.write('%stub\nf e 5\na d 6\nc b 1')
        text_file.close()

        self.stub_data_3 = 'stub_3.txt'
        text_file = open(self.stub_data_3, "w")
        text_file.write('%stub\n1 3 a\n4 5 b\n0 2 e')
        text_file.close()

    def tearDown(self):
        remove(self.stub_data_1)
        remove(self.stub_data_2)
        remove(self.stub_data_3)

    def test_unlabeled_unweighted(self):
        adj = parser.parse_tsv(self.stub_data_1)
        self.assertEqual(sum(adj.indices == np.array([2, 3, 0, 1, 5, 4])), 6)
        self.assertEqual(sum(adj.indptr == np.array([0, 1, 2, 3, 4, 5, 6])), 7)
        self.assertEqual(sum(adj.data == np.array([1, 1, 1, 1, 1, 1])), 6)

    def test_labeled_weighted(self):
        adj, labels = parser.parse_tsv(self.stub_data_2)
        self.assertEqual(sum(adj.indices == np.array([3, 2, 1, 0, 5, 4])), 6)
        self.assertEqual(sum(adj.indptr == np.array([0, 1, 2, 3, 4, 5, 6])), 7)
        self.assertEqual(sum(adj.data == np.array([6, 1, 1, 6, 5, 5])), 6)
        self.assertEqual(sum(labels == np.array(['a', 'b', 'c', 'd', 'e', 'f'])), 6)

    def test_poor_format(self):
        self.assertRaises(ValueError, parser.parse_tsv, self.stub_data_3)

    def test_fast_parse(self):
        adj = parser.fast_parse_tsv(self.stub_data_1)
        self.assertEqual(sum(adj.indices == np.array([2, 3, 0, 1, 5, 4])), 6)
        self.assertEqual(sum(adj.indptr == np.array([0, 1, 2, 3, 4, 5, 6])), 7)
        self.assertEqual(sum(adj.data == np.array([1, 1, 1, 1, 1, 1])), 6)


