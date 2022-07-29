"""classification module"""
from sknetwork.classification.base import BaseClassifier
from sknetwork.classification.diffusion import DiffusionClassifier
from sknetwork.classification.knn import KNN
from sknetwork.classification.metrics import get_accuracy_score, get_confusion_matrix, get_f1_scores
from sknetwork.classification.pagerank import PageRankClassifier
from sknetwork.classification.propagation import Propagation
