"""link prediction module"""
from sknetwork.linkpred.first_order import CommonNeighbors, JaccardIndex, SaltonIndex, SorensenIndex, HubPromotedIndex,\
    HubDepressedIndex, AdamicAdar, ResourceAllocation, PreferentialAttachment
from sknetwork.linkpred.postprocessing import is_edge, whitened_sigmoid
