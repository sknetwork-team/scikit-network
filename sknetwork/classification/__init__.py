"""classification module"""
from sknetwork.classification.base import BaseClassifier, BaseBiClassifier
from sknetwork.classification.knn import KNN, BiKNN
from sknetwork.classification.metrics import accuracy_score
from sknetwork.classification.pagerank import BiPageRankClassifier, PageRankClassifier
from sknetwork.classification.diffusion import DiffusionClassifier, BiDiffusionClassifier, DirichletClassifier,\
    BiDirichletClassifier
from sknetwork.classification.propagation import BiPropagation, Propagation
