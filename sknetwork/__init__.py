#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level package for scikit-network"""

__author__ = """scikit-network team"""
__email__ = "bonald@enst.fr"
__version__ = '0.6.1'

import numpy as np
try:
    from numba import njit, prange, types
    from numba.typed import Dict as TypedDict
    is_numba_available = True
except ImportError:
    def njit(*args, **kwargs):
        if len(args) > 0:
            if callable(args[0]):
                return args[0]
        else:
            def __wrapper__(func):
                return func
            return __wrapper__
    prange = range
    is_numba_available = False
    types = np

    class TypedDict(dict):
        @staticmethod
        def empty(**kwargs):
            pass


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
from sknetwork.utils.randomized_matrix_factorization import *
from sknetwork.utils.preprocessing import *
from sknetwork.loader import *
