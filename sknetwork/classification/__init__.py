"""classification module"""
from sknetwork.classification.base import BaseClassifier, BaseBiClassifier
from sknetwork.classification.knn import KNN, BiKNN
from sknetwork.classification.pagerank import BiPageRankClassifier, PageRankClassifier, CoPageRankClassifier
from sknetwork.classification.diffusion import BiDiffusionClassifier, DiffusionClassifier
from sknetwork.classification.propagation import BiPropagation, Propagation
