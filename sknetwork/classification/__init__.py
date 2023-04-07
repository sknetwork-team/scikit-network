"""classification module"""
from sknetwork.classification.base import BaseClassifier
from sknetwork.classification.diffusion import DiffusionClassifier
from sknetwork.classification.knn import NNClassifier
from sknetwork.classification.metrics import get_accuracy_score, get_confusion_matrix, get_f1_score, get_f1_scores, \
    get_average_f1_score
from sknetwork.classification.pagerank import PageRankClassifier
from sknetwork.classification.propagation import Propagation
