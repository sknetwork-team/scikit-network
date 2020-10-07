"""data module"""
from sknetwork.data.load import load_netset, load_konect, clear_data_home, get_data_home, save, load
from sknetwork.data.models import *
from sknetwork.data.parse import parse_edge_list, load_labels, parse_graphml, parse_adjacency_list, load_edge_list
from sknetwork.data.toy_graphs import *
