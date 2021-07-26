"""classification module"""
from sknetwork.classification.base import BaseClassifier
from sknetwork.classification.diffusion import DiffusionClassifier, DirichletClassifier
from sknetwork.classification.knn import KNN
from sknetwork.classification.metrics import accuracy_score
from sknetwork.classification.pagerank import PageRankClassifier
from sknetwork.classification.propagation import Propagation
