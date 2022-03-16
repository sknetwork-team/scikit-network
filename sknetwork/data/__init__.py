"""data module"""
from sknetwork.data.load import load_netset, load_konect, clear_data_home, get_data_home, save, load
from sknetwork.data.models import *
from sknetwork.data.parse import from_csv, from_csv_adjacency, from_graphml, from_coo_format, from_edge_list
from sknetwork.data.toy_graphs import *
