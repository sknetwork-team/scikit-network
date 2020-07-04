"""linalg module"""
from sknetwork.linalg.auto_mode import auto_solver
from sknetwork.linalg.eig_solver import EigSolver, LanczosEig, HalkoEig
from sknetwork.linalg.operators import RegularizedAdjacency, LaplacianOperator, NormalizedAdjacencyOperator,\
    CoNeighborOperator
from sknetwork.linalg.normalization import diag_pinv, normalize
from sknetwork.linalg.randomized_methods import safe_sparse_dot, randomized_eig, randomized_svd
from sknetwork.linalg.sparse_lowrank import SparseLR
from sknetwork.linalg.svd_solver import SVDSolver, LanczosSVD, HalkoSVD
from sknetwork.linalg.polynome import Polynome
