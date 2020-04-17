"""embedding module"""
from sknetwork.embedding.base import BaseEmbedding
from sknetwork.embedding.metrics import cosine_modularity
from sknetwork.embedding.spectral import Spectral, BiSpectral
from sknetwork.embedding.spring import Spring
from sknetwork.embedding.svd import SVD, GSVD
