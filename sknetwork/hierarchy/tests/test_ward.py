#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

import unittest

from sknetwork.hierarchy import Ward
from sknetwork.embedding import Spectral
from sknetwork.data.test_graphs import test_graph


class TestWard(unittest.TestCase):

    def test_options(self):
        adjacency = test_graph()
        ward = Ward(embedding_method=Spectral(3))
        dendrogram = ward.fit_transform(adjacency)
        self.assertEqual(dendrogram.shape, (adjacency.shape[0] - 1, 4))
