"""hierarchy module"""
from sknetwork.hierarchy.base import BaseHierarchy
from sknetwork.hierarchy.cuts import cut_straight, cut_balanced
from sknetwork.hierarchy.metrics import dasgupta_score, tree_sampling_divergence
from sknetwork.hierarchy.paris import Paris, BiParis
from sknetwork.hierarchy.ward import Ward, BiWard
