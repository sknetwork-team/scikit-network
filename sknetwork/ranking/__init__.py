"""ranking module"""
from sknetwork.ranking.base import BaseRanking
from sknetwork.ranking.closeness import Closeness
from sknetwork.ranking.diffusion import Diffusion, BiDiffusion
from sknetwork.ranking.harmonic import Harmonic
from sknetwork.ranking.hits import HITS
from sknetwork.ranking.pagerank import PageRank, BiPageRank, CoPageRank
