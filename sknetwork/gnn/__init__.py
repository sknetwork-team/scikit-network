"""gnn module"""
from sknetwork.gnn.base_gnn import BaseGNNClassifier
from sknetwork.gnn.gnn_classifier import GNNClassifier
from sknetwork.gnn.layers import GCNConv
from sknetwork.gnn.activation import *
from sknetwork.gnn.loss import *
from sknetwork.gnn.optimizer import GD, ADAM, optimizer_factory
from sknetwork.gnn.utils import *
