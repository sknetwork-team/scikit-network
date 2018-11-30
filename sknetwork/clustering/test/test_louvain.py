# -*- coding: utf-8 -*-
# test for louvain.py
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Quentin Lutz <qlutz@enst.fr>
#
# This file is part of Scikit-network.

import unittest
from sknetwork.clustering.louvain import Louvain
from scipy.sparse import identity


class TestLouvainClustering(unittest.TestCase):

    def setUp(self):
        self.louvain_basic = Louvain()

    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            self.louvain_basic.fit(identity(1))

        with self.assertRaises(TypeError):
            self.louvain_basic.fit(identity(2, format='csr'), node_weights=1)

    def test_unknown_options(self):
        with self.assertRaises(ValueError):
            self.louvain_basic.fit(identity(2, format='csr'), node_weights='unknown')

    def test_single_node_graph(self):
        self.assertEqual(self.louvain_basic.fit(identity(1, format='csr')).labels_, [0])
