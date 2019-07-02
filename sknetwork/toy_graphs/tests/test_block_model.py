# -*- coding: utf-8 -*-
# tests for graph_data.py
# authors: Quentin Lutz <qlutz@enst.fr>

import unittest
from sknetwork.toy_graphs.block_model import block_model


class TestGraphImport(unittest.TestCase):

    def setUp(self):
        pass

    def test_errors(self):
        with self.assertRaises(TypeError):
            block_model('foo')

        with self.assertRaises(ValueError):
            block_model(-1)

        with self.assertRaises(ValueError):
            block_model(10)
