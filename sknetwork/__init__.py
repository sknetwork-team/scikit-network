#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level package for scikit-network"""

__author__ = """scikit-network team"""
__email__ = "bonald@enst.fr"
__version__ = '0.2.0'

from scipy.sparse.csgraph import *
from sknetwork.toy_graphs import *
from sknetwork.loader import *
from sknetwork.clustering import *
from sknetwork.hierarchy import *
from sknetwork.embedding import *