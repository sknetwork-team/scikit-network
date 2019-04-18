#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for parser.py"""


import unittest
from os import remove
from sknetwork.loader import parser


class TestTSVParser(unittest.TestCase):

    def setUp(self):
        self.stub_data_1 = 'stub_1.txt'
        text_file = open(self.stub_data_1, "w")
        text_file.write('%stub\n1 3\n4 5\n0 2')
        text_file.close()

        self.stub_data_2 = 'stub_2.txt'
        text_file = open(self.stub_data_2, "w")
        text_file.write('%stub\nf, e, 5\na, d, 6\nc, b, 1')
        text_file.close()

        self.stub_data_3 = 'stub_3.txt'
        text_file = open(self.stub_data_3, "w")
        text_file.write('%stub\n1 3 a\n4 5 b\n0 2 e')
        text_file.close()

        self.stub_data_4 = 'stub_4.txt'
        text_file = open(self.stub_data_4, "w")
        text_file.write('%stub\n14 31\n42 50\n0 12')
        text_file.close()

    def tearDown(self):
        remove(self.stub_data_1)
        remove(self.stub_data_2)
        remove(self.stub_data_3)
        remove(self.stub_data_4)

    def test_unlabeled_unweighted(self):
        adj, labels = parser.parse_tsv(self.stub_data_1)
        self.assertEqual(sum(adj.indices == [2, 3, 0, 1, 5, 4]), 6)
        self.assertEqual(sum(adj.indptr == [0, 1, 2, 3, 4, 5, 6]), 7)
        self.assertEqual(sum(adj.data == [1, 1, 1, 1, 1, 1]), 6)
        self.assertIsNone(labels)

    def test_labeled_weighted(self):
        adj, labels = parser.parse_tsv(self.stub_data_2)
        self.assertEqual(sum(adj.indices == [4, 3, 5, 1, 0, 2]), 6)
        self.assertEqual(sum(adj.indptr == [0, 1, 2, 3, 4, 5, 6]), 7)
        self.assertEqual(sum(adj.data == [1, 6, 5, 6, 1, 5]), 6)
        self.assertEqual(sum(labels == [' b', ' d', ' e', 'a', 'c', 'f']), 6)

    def test_auto_reindex(self):
        adj, labels = parser.parse_tsv(self.stub_data_4)
        self.assertEqual(sum(adj.indices == [1, 0, 3, 2, 5, 4]), 6)
        self.assertEqual(sum(adj.indptr == [0, 1, 2, 3, 4, 5, 6]), 7)
        self.assertEqual(sum(adj.data == [1, 1, 1, 1, 1, 1]), 6)
        self.assertEqual(sum(labels == [0, 12, 14, 31, 42, 50]), 6)

    def test_poor_format(self):
        self.assertRaises(ValueError, parser.parse_tsv, self.stub_data_3)
