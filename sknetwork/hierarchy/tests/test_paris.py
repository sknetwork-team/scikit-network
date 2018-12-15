# -*- coding: utf-8 -*-
# test for louvain.py
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Thomas Bonald <tbonald@enst.fr>
#
# This file is part of Scikit-network.

import unittest
from sknetwork.hierarchy.paris import Paris
from scipy.sparse import identity


class TestParis(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()

    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.paris.fit(identity(1))

        with self.assertRaises(TypeError):
            self.paris.fit(identity(2, format='csr'), node_weights=1)

    def test_unknown_options(self):
        with self.assertRaises(ValueError):
            self.paris.fit(identity(2, format='csr'), node_weights='unknown')

    def test_single_node_graph(self):
        self.assertEqual(self.paris.fit(identity(1, format='csr')).dendrogram_.shape[0], 0)
