"""Module of linear algebra."""
from sknetwork.linalg.basics import safe_sparse_dot
from sknetwork.linalg.eig_solver import EigSolver, LanczosEig
from sknetwork.linalg.normalization import diag_pinv, normalize
from sknetwork.linalg.operators import Regularizer, Laplacian, Normalizer, CoNeighbor
from sknetwork.linalg.polynome import Polynome
from sknetwork.linalg.sparse_lowrank import SparseLR
from sknetwork.linalg.svd_solver import SVDSolver, LanczosSVD
