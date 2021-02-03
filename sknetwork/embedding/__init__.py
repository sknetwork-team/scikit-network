"""embedding module"""
from sknetwork.embedding.base import BaseEmbedding, BaseBiEmbedding
from sknetwork.embedding.force_atlas import ForceAtlas
from sknetwork.embedding.louvain_embedding import BiLouvainEmbedding, LouvainEmbedding
from sknetwork.embedding.louvain_hierarchy import BiHLouvainEmbedding, HLouvainEmbedding
from sknetwork.embedding.metrics import cosine_modularity
from sknetwork.embedding.random_projection import BiRandomProjection, RandomProjection
from sknetwork.embedding.spectral import Spectral, BiSpectral, LaplacianEmbedding
from sknetwork.embedding.spring import Spring
from sknetwork.embedding.svd import SVD, GSVD, PCA
