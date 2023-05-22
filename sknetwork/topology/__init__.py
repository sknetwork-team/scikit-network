"""Module on topology."""
from sknetwork.topology.cliques import count_cliques
from sknetwork.topology.core import get_core_decomposition
from sknetwork.topology.triangles import count_triangles, get_clustering_coefficient
from sknetwork.topology.structure import *
from sknetwork.topology.weisfeiler_lehman import color_weisfeiler_lehman, are_isomorphic
