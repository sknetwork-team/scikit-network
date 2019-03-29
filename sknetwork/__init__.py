#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level package for scikit-network"""

__author__ = """scikit-network team"""
__email__ = "bonald@enst.fr"
__version__ = '0.2.0'

from scipy.sparse.csgraph import *
from sknetwork.toy_graphs.graph_data import *
from sknetwork.loader.parser import *
from sknetwork.clustering.louvain import *
from sknetwork.clustering.bilouvain import *
from sknetwork.clustering.metrics import *
from sknetwork.hierarchy.paris import *
from sknetwork.hierarchy.metrics import *
from sknetwork.embedding.gsvd import *
from sknetwork.embedding.spectral import *
from sknetwork.embedding.metrics import *
from sknetwork.embedding.randomized_matrix_factorization import *
