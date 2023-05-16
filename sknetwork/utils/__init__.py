"""utils module"""
from sknetwork.data import Bunch
from sknetwork.utils.check import is_symmetric
from sknetwork.utils.format import *
from sknetwork.utils.membership import get_membership, from_membership
from sknetwork.utils.neighbors import get_neighbors, get_degrees, get_weights
from sknetwork.utils.simplex import projection_simplex, projection_simplex_array, projection_simplex_csr
from sknetwork.utils.tfidf import get_tfidf

