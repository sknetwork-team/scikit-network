.. _gnn:

GNN
***

Graph Neural Network.

Classifier
----------

The attribute ``labels_``  assigns a label to each node of the graph.

.. autoclass:: sknetwork.gnn.GNNClassifier

Convolution layers
------------------

.. autoclass:: sknetwork.gnn.Convolution

Activation functions
--------------------

.. autoclass:: sknetwork.gnn.BaseActivation
.. autoclass:: sknetwork.gnn.ReLu
.. autoclass:: sknetwork.gnn.Sigmoid
.. autoclass:: sknetwork.gnn.Softmax

Loss functions
--------------

.. autoclass:: sknetwork.gnn.BaseLoss
.. autoclass:: sknetwork.gnn.CrossEntropy
.. autoclass:: sknetwork.gnn.BinaryCrossEntropy

Optimizers
----------

.. autoclass:: sknetwork.gnn.BaseOptimizer
.. autoclass:: sknetwork.gnn.ADAM
.. autoclass:: sknetwork.gnn.GD
