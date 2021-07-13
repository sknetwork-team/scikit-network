#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from sknetwork.data.test_graphs import test_graph, test_digraph, test_bigraph
from sknetwork.embedding import Spectral
from sknetwork.hierarchy import Ward


class TestWard(unittest.TestCase):

    def test_options(self):
        ward = Ward()
        ward_options = Ward(embedding_method=Spectral(3), co_cluster=True)
        for algo in [ward, ward_options]:
            for input_matrix in [test_graph(), test_digraph(), test_bigraph()]:
                dendrogram = algo.fit_transform(input_matrix)
                self.assertEqual(dendrogram.shape, (input_matrix.shape[0] - 1, 4))
                if algo.co_cluster:
                    self.assertEqual(algo.dendrogram_full_.shape, (sum(input_matrix.shape) - 1, 4))
