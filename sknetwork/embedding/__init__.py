"""embedding module"""
from sknetwork.embedding.base import BaseEmbedding
from sknetwork.embedding.force_atlas import ForceAtlas
from sknetwork.embedding.louvain_embedding import LouvainEmbedding
from sknetwork.embedding.random_projection import RandomProjection
from sknetwork.embedding.spectral import Spectral
from sknetwork.embedding.spring import Spring
from sknetwork.embedding.svd import SVD, GSVD, PCA
