"""classification module"""
from sknetwork.classification.base import BaseClassifier
from sknetwork.classification.knn import KNN, BiKNN
from sknetwork.classification.pagerank import BiPageRankClassifier, PageRankClassifier
from sknetwork.classification.diffusion import BiDiffusionClassifier, DiffusionClassifier
