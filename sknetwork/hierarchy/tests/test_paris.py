#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2019
@author: Thomas Bonald <bonald@enst.fr>
@author: Quentin Lutz <qlutz@enst.fr>
"""

import unittest

from sknetwork.hierarchy import Paris
from sknetwork.data.test_graphs import test_graph


class TestParis(unittest.TestCase):

    def test_options(self):
        paris = Paris(weights='uniform')
        adjacency = test_graph()
        dendrogram = paris.fit_transform(adjacency)
        n = adjacency.shape[0]
        self.assertEqual(dendrogram.shape, (n - 1, 4))
