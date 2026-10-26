#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level package for scikit-network"""
import importlib.metadata as _metadata
try:
    __metadata__ = _metadata.metadata("scikit-network")
except:
    __metadata__ = {}

__author__  = __metadata__.get("Author", "Scikit-Network team")
__email__   = __metadata__.get("Maintainer-email", "thomas.bonald@telecom-paris.fr")
__version__ = __metadata__.get("Version", "0.0.0")

import sknetwork.topology
import sknetwork.path
import sknetwork.classification
import sknetwork.clustering
import sknetwork.embedding
import sknetwork.hierarchy
import sknetwork.linalg
import sknetwork.linkpred
import sknetwork.ranking
import sknetwork.data
import sknetwork.utils
import sknetwork.visualization
import sknetwork.gnn
