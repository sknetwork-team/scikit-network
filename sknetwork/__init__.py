# -*- coding: utf-8 -*-

"""Top-level package for scikit-network."""

__author__ = """Bertrand Charpentier"""
__email__ = 'bertrand.charpentier@live.fr'
__version__ = '0.1.1'

from .hierarchical_clustering.agglomerative_clustering import linkage_clustering
from .hierarchical_clustering.metrics import hierarchical_cost
from .embedding.forward_backward_embedding import ForwardBackwardEmbedding

