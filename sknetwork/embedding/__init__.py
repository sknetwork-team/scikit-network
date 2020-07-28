"""embedding module"""
from sknetwork.embedding.base import BaseEmbedding, BaseBiEmbedding
from sknetwork.embedding.metrics import cosine_modularity
from sknetwork.embedding.spectral import Spectral, BiSpectral, LaplacianEmbedding
from sknetwork.embedding.spring import Spring
from sknetwork.embedding.force_atlas import ForceAtlas2
from sknetwork.embedding.svd import SVD, GSVD
