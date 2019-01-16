# -*- coding: utf-8 -*-
# test for metrics.py
#
# Copyright 2018 Scikit-network Developers.
# Copyright 2018 Bertrand Charpentier <bertrand.charpentier@live.fr>
#
# This file is part of Scikit-network.
#

import unittest
from sknetwork.hierarchy.paris import Paris

class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.paris = Paris()
