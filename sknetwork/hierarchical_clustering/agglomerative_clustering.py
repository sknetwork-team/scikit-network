# import numpy as np
# import networkx as nx
#
#
# _AFFINITY = {'unitary', 'weighted'}
# _LINKAGE = {'single', 'average', 'complete', 'modular'}
#
#
# def agglomerative_clustering(graph, affinity='weighted', linkage='modular', f=lambda l: - np.log(l), check=True):
#     """
#     Compute a hierarchy fo the :attr:`graph` with the given :attr:`linkage`. The graph can be weighted or unweighted
#     :param (graph) graph: A NetworkX graph
#     :param affinity: 'weighted' or 'unitary'. 'weighted' takes the attribute 'weight' on the edges
#     whereas 'unitary' assign
#     :param (string) linkage: 'single', 'average', 'complete' or 'modular'.
#     :param (function) f: transformation of the linkage into a distance.
#     :param (bool) check: a beautiful number.
#     :return: :attr:`dendrogram` containing the successive merges and their respective distance
#     :raise ValueError: if :attr:`affinity` or :attr:`linkage` is not known.
#     :raise KeyError: if all the edges of the :attr:`graph` do not have the attribute 'weight'.
#     Example
#     >>> dendrogram = agglomerative_clustering(graph)
#     """
#
#     if affinity not in _AFFINITY:
#         raise ValueError("Unknown affinity type %s."
#                          "Valid options are %s" % (affinity, _AFFINITY.keys()))
#
#     if linkage not in _LINKAGE:
#         raise ValueError("Unknown linkage type %s."
#                          "Valid options are %s" % (linkage, _LINKAGE.keys()))
#
#     graph_copy = graph.copy()
#
#     if check:
#
#         graph_copy = nx.convert_node_labels_to_integers(graph_copy)
#
#         if affinity == 'unitary':
#             for e in graph_copy.edges:
#                 graph_copy.add_edge(e[0], e[1], weight=1)
#
#         n_edges = len(list(graph_copy.edges()))
#         n_weighted_edges = len(nx.get_edge_attributes(graph_copy, 'weight'))
#         if affinity == 'weighted' and not n_weighted_edges == n_edges:
#             raise KeyError("%s edges among %s do not have the attribute/key \'weigth\'."
#                            % (n_edges - n_weighted_edges, n_edges))
#
#     if linkage == 'single':
#         dendrogram = single_linkage_hierarchy(graph_copy, f)
#     elif linkage == 'average':
#         dendrogram = average_linkage_hierarchy(graph_copy, f)
#     elif linkage == 'complete':
#         dendrogram = complete_linkage_hierarchy(graph_copy, f)
#     elif linkage == 'modular':
#         dendrogram = modular_linkage_hierarchy(graph_copy, f)
#
#     return reorder_dendrogram(dendrogram)
#
#
# def single_linkage_hierarchy(graph, f):
#     remaining_nodes = set(graph.nodes())
#     n_nodes = len(remaining_nodes)
#
#     cluster_size = {u: 1 for u in range(n_nodes)}
#     connected_components = []
#     dendrogram = []
#     u = n_nodes
#
#     while n_nodes > 0:
#         for new_node in remaining_nodes:
#             chain = [new_node]
#             break
#         while chain:
#             a = chain.pop()
#             linkage_max = - float("inf")
#             b = -1
#             neighbors_a = list(graph.neighbors(a))
#             for v in neighbors_a:
#                 if v != a:
#                     linkage = float(graph[a][v]['weight'])
#                     if linkage > linkage_max:
#                         b = v
#                         linkage_max = linkage
#                     elif linkage == linkage_max:
#                         b = min(b, v)
#             linkage = linkage_max
#             if chain:
#                 c = chain.pop()
#                 if b == c:
#                     dendrogram.append([a, b, f(linkage), cluster_size[a] + cluster_size[b]])
#                     graph.add_node(u)
#                     remaining_nodes.add(u)
#                     neighbors_a = list(graph.neighbors(a))
#                     neighbors_b = list(graph.neighbors(b))
#                     for v in neighbors_a:
#                         graph.add_edge(u, v, weight=graph[a][v]['weight'])
#                     for v in neighbors_b:
#                         if graph.has_edge(u, v):
#                             graph[u][v]['weight'] = max(graph[b][v]['weight'], graph[u][v]['weight'])
#                         else:
#                             graph.add_edge(u, v, weight=graph[b][v]['weight'])
#                     graph.remove_node(a)
#                     remaining_nodes.remove(a)
#                     graph.remove_node(b)
#                     remaining_nodes.remove(b)
#                     n_nodes -= 1
#                     cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
#                     u += 1
#                 else:
#                     chain.append(c)
#                     chain.append(a)
#                     chain.append(b)
#             elif b >= 0:
#                 chain.append(a)
#                 chain.append(b)
#             else:
#                 connected_components.append((a, cluster_size[a]))
#                 graph.remove_node(a)
#                 cluster_size.pop(a)
#                 n_nodes -= 1
#
#     a, cluster_size = connected_components.pop()
#     for b, t in connected_components:
#         cluster_size += t
#         dendrogram.append([a, b, float("inf"), cluster_size])
#         a = u
#         u += 1
#
#     return np.array(dendrogram)
#
#
# def average_linkage_hierarchy(graph, f):
#     remaining_nodes = set(graph.nodes())
#     n_nodes = len(remaining_nodes)
#
#     wtot = 0
#     for (u, v) in graph.edges():
#         weight = graph[u][v]['weight']
#         wtot += 2 * weight
#     n_nodes2_wtot = n_nodes * n_nodes / wtot
#     cluster_size = {u: 1 for u in range(n_nodes)}
#     connected_components = []
#     dendrogram = []
#     u = n_nodes
#
#     while n_nodes > 0:
#         for new_node in remaining_nodes:
#             chain = [new_node]
#             break
#         while chain:
#             a = chain.pop()
#             linkage_max = - float("inf")
#             b = -1
#             neighbors_a = list(graph.neighbors(a))
#             for v in neighbors_a:
#                 if v != a:
#                     linkage = n_nodes2_wtot * float(graph[a][v]['weight'])/(cluster_size[a]*cluster_size[v])
#                     if linkage > linkage_max:
#                         b = v
#                         linkage_max = linkage
#                     elif linkage == linkage_max:
#                         b = min(b, v)
#             linkage = linkage_max
#             if chain:
#                 c = chain.pop()
#                 if b == c:
#                     dendrogram.append([a, b, f(linkage), cluster_size[a] + cluster_size[b]])
#                     graph.add_node(u)
#                     remaining_nodes.add(u)
#                     neighbors_a = list(graph.neighbors(a))
#                     neighbors_b = list(graph.neighbors(b))
#                     for v in neighbors_a:
#                         graph.add_edge(u, v, weight=graph[a][v]['weight'])
#                     for v in neighbors_b:
#                         if graph.has_edge(u, v):
#                             graph[u][v]['weight'] += graph[b][v]['weight']
#                         else:
#                             graph.add_edge(u, v, weight=graph[b][v]['weight'])
#                     graph.remove_node(a)
#                     remaining_nodes.remove(a)
#                     graph.remove_node(b)
#                     remaining_nodes.remove(b)
#                     n_nodes -= 1
#                     cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
#                     u += 1
#                 else:
#                     chain.append(c)
#                     chain.append(a)
#                     chain.append(b)
#             elif b >= 0:
#                 chain.append(a)
#                 chain.append(b)
#             else:
#                 connected_components.append((a, cluster_size[a]))
#                 graph.remove_node(a)
#                 cluster_size.pop(a)
#                 n_nodes -= 1
#
#     a, cluster_size = connected_components.pop()
#     for b, t in connected_components:
#         cluster_size += t
#         dendrogram.append([a, b, float("inf"), cluster_size])
#         a = u
#         u += 1
#
#     return np.array(dendrogram)
#
#
# def complete_linkage_hierarchy(graph, f):
#     remaining_nodes = set(graph.nodes())
#     n_nodes = len(remaining_nodes)
#
#     cluster_size = {u: 1 for u in range(n_nodes)}
#     connected_components = []
#     dendrogram = []
#     u = n_nodes
#
#     while n_nodes > 0:
#         for new_node in remaining_nodes:
#             chain = [new_node]
#             break
#         while chain:
#             a = chain.pop()
#             linkage_max = - float("inf")
#             b = -1
#             neighbors_a = list(graph.neighbors(a))
#             for v in neighbors_a:
#                 if v != a:
#                     linkage = float(graph[a][v]['weight'])
#                     if linkage > linkage_max:
#                         b = v
#                         linkage_max = linkage
#                     elif linkage == linkage_max:
#                         b = min(b, v)
#             linkage = linkage_max
#             if chain:
#                 c = chain.pop()
#                 if b == c:
#                     dendrogram.append([a, b, f(linkage), cluster_size[a] + cluster_size[b]])
#                     graph.add_node(u)
#                     remaining_nodes.add(u)
#                     neighbors_a = list(graph.neighbors(a))
#                     neighbors_b = list(graph.neighbors(b))
#                     for v in neighbors_a:
#                         graph.add_edge(u, v, weight=graph[a][v]['weight'])
#                     for v in neighbors_b:
#                         if graph.has_edge(u, v):
#                             graph[u][v]['weight'] = min(graph[b][v]['weight'], graph[u][v]['weight'])
#                         else:
#                             graph.add_edge(u, v, weight=graph[b][v]['weight'])
#                     graph.remove_node(a)
#                     remaining_nodes.remove(a)
#                     graph.remove_node(b)
#                     remaining_nodes.remove(b)
#                     n_nodes -= 1
#                     cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
#                     u += 1
#                 else:
#                     chain.append(c)
#                     chain.append(a)
#                     chain.append(b)
#             elif b >= 0:
#                 chain.append(a)
#                 chain.append(b)
#             else:
#                 connected_components.append((a, cluster_size[a]))
#                 graph.remove_node(a)
#                 cluster_size.pop(a)
#                 n_nodes -= 1
#
#     a, cluster_size = connected_components.pop()
#     for b, t in connected_components:
#         cluster_size += t
#         dendrogram.append([a, b, float("inf"), cluster_size])
#         a = u
#         u += 1
#
#     return np.array(dendrogram)
#
#
# def modular_linkage_hierarchy(graph, f):
#     remaining_nodes = set(graph.nodes())
#     n_nodes = len(remaining_nodes)
#
#     w = {u: 0 for u in range(n_nodes)}
#     wtot = 0
#     for (u, v) in graph.edges():
#         weight = graph[u][v]['weight']
#         w[u] += weight
#         w[v] += weight
#         wtot += 2 * weight
#     cluster_size = {u: 1 for u in range(n_nodes)}
#     connected_components = []
#     dendrogram = []
#     u = n_nodes
#
#     while n_nodes > 0:
#         for new_node in remaining_nodes:
#             chain = [new_node]
#             break
#         while chain:
#             a = chain.pop()
#             d_min = float("inf")
#             b = -1
#             neighbors_a = list(graph.neighbors(a))
#             for v in neighbors_a:
#                 if v != a:
#                     linkage = wtot * float(graph[a][v]['weight'])/(w[a]*w[v])
#                     if linkage > linkage_max:
#                         b = v
#                         linkage_max = linkage
#                     elif linkage == linkage_max:
#                         b = min(b, v)
#             linkage = linkage_max
#             if chain:
#                 c = chain.pop()
#                 if b == c:
#                     dendrogram.append([a, b, f(linkage), cluster_size[a] + cluster_size[b]])
#                     graph.add_node(u)
#                     remaining_nodes.add(u)
#                     neighbors_a = list(graph.neighbors(a))
#                     neighbors_b = list(graph.neighbors(b))
#                     for v in neighbors_a:
#                         graph.add_edge(u, v, weight=graph[a][v]['weight'])
#                     for v in neighbors_b:
#                         if graph.has_edge(u, v):
#                             graph[u][v]['weight'] += graph[b][v]['weight']
#                         else:
#                             graph.add_edge(u, v, weight=graph[b][v]['weight'])
#                     graph.remove_node(a)
#                     remaining_nodes.remove(a)
#                     graph.remove_node(b)
#                     remaining_nodes.remove(b)
#                     n_nodes -= 1
#                     w[u] = w.pop(a) + w.pop(b)
#                     cluster_size[u] = cluster_size.pop(a) + cluster_size.pop(b)
#                     u += 1
#                 else:
#                     chain.append(c)
#                     chain.append(a)
#                     chain.append(b)
#             elif b >= 0:
#                 chain.append(a)
#                 chain.append(b)
#             else:
#                 connected_components.append((a, cluster_size[a]))
#                 graph.remove_node(a)
#                 w.pop(a)
#                 cluster_size.pop(a)
#                 n_nodes -= 1
#
#     a, cluster_size = connected_components.pop()
#     for b, t in connected_components:
#         cluster_size += t
#         dendrogram.append([a, b, float("inf"), cluster_size])
#         a = u
#         u += 1
#
#     return np.array(dendrogram)
#
#
# def reorder_dendrogram(D):
#     n = np.shape(D)[0] + 1
#     order = np.zeros((2, n - 1), float)
#     order[0] = range(n - 1)
#     order[1] = np.array(D)[:, 2]
#     index = np.lexsort(order)
#     nindex = {i: i for i in range(n)}
#     nindex.update({n + index[t]: n + t for t in range(n - 1)})
#     return np.array([[nindex[int(D[t][0])], nindex[int(D[t][1])], D[t][2], D[t][3]] for t in range(n - 1)])[index, :]
