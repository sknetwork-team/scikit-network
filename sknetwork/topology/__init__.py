"""Module on topology."""
from sknetwork.topology.kcliques import Cliques
from sknetwork.topology.kcore import CoreDecomposition
from sknetwork.topology.triangles import Triangles

from sknetwork.topology.dag import DAG
from sknetwork.topology.structure import is_acyclic, is_bipartite, is_connected, get_largest_connected_component, \
    get_connected_components
from sknetwork.topology.weisfeiler_lehman import WeisfeilerLehman, are_isomorphic
