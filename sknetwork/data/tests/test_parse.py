#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for parse.py"""

import unittest
from os import remove

from sknetwork.data import parse


class TestParser(unittest.TestCase):

    def test_unlabeled_unweighted(self):
        self.stub_data_1 = 'stub_1.txt'
        with open(self.stub_data_1, "w") as text_file:
            text_file.write('%stub\n1 3\n4 5\n0 2')
        graph = parse.load_edge_list(self.stub_data_1)
        adjacency = graph.adjacency
        self.assertTrue((adjacency.indices == [2, 3, 0, 1, 5, 4]).all())
        self.assertTrue((adjacency.indptr == [0, 1, 2, 3, 4, 5, 6]).all())
        self.assertTrue((adjacency.data == [1, 1, 1, 1, 1, 1]).all())
        remove(self.stub_data_1)

    def test_labeled_weighted(self):
        self.stub_data_2 = 'stub_2.txt'
        with open(self.stub_data_2, "w") as text_file:
            text_file.write('%stub\nf, e, 5\na, d, 6\nc, b, 1')
        graph = parse.load_edge_list(self.stub_data_2)
        adjacency = graph.adjacency
        names = graph.names
        self.assertTrue((adjacency.indices == [4, 3, 5, 1, 0, 2]).all())
        self.assertTrue((adjacency.indptr == [0, 1, 2, 3, 4, 5, 6]).all())
        self.assertTrue((adjacency.data == [1, 6, 5, 6, 1, 5]).all())
        self.assertTrue((names == [' b', ' d', ' e', 'a', 'c', 'f']).all())
        remove(self.stub_data_2)

    def test_auto_reindex(self):
        self.stub_data_4 = 'stub_4.txt'
        with open(self.stub_data_4, "w") as text_file:
            text_file.write('%stub\n14 31\n42 50\n0 12')
        graph = parse.load_edge_list(self.stub_data_4)
        adjacency = graph.adjacency
        names = graph.names
        self.assertTrue((adjacency.indices == [1, 0, 3, 2, 5, 4]).all())
        self.assertTrue((adjacency.indptr == [0, 1, 2, 3, 4, 5, 6]).all())
        self.assertTrue((adjacency.data == [1, 1, 1, 1, 1, 1]).all())
        self.assertTrue((names == [0, 12, 14, 31, 42, 50]).all())
        remove(self.stub_data_4)

    def test_wrong_format_fast(self):
        self.stub_data_3 = 'stub_3.txt'
        with open(self.stub_data_3, "w") as text_file:
            text_file.write('%stub\n1 3 a\n4 5 b\n0 2 e')
        self.assertRaises(ValueError, parse.load_edge_list, self.stub_data_3)
        remove(self.stub_data_3)

    def test_wrong_format_slow(self):
        self.stub_data_3 = 'stub_3.txt'
        with open(self.stub_data_3, "w") as text_file:
            text_file.write('%stub\n1 3 a\n4 5 b\n0 2 e')
        self.assertRaises(ValueError, parse.load_edge_list, self.stub_data_3, fast_format=False)
        remove(self.stub_data_3)

    def test_graphml_basic(self):
        self.stub_data_5 = 'stub_5.graphml'
        with open(self.stub_data_5, "w") as graphml_file:
            graphml_file.write("""<?xml version='1.0' encoding='utf-8'?>
                                    <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
                                    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                                    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
                                    http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
                                    <key id="d0" for="edge" attr.name="weight" attr.type="int"/>
                                    <graph edgedefault="directed">
                                    <node id="node1"/>
                                    <node id="node2"/>
                                    <edge source="node1" target="node2">
                                      <data key="d0">1</data>
                                    </edge></graph></graphml>""")
        graph = parse.load_graphml(self.stub_data_5)
        adjacency = graph.adjacency
        names = graph.names
        self.assertTrue((adjacency.indices == [1]).all())
        self.assertTrue((adjacency.indptr == [0, 1, 1]).all())
        self.assertTrue((adjacency.data == [1]).all())
        self.assertTrue((names == ['node1', 'node2']).all())
        remove(self.stub_data_5)

    def test_graphml_refined(self):
        self.stub_data_6 = 'stub_6.graphml'
        with open(self.stub_data_6, "w") as graphml_file:
            graphml_file.write("""<?xml version='1.0' encoding='utf-8'?>
                                    <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
                                    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                                    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
                                    http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
                                      <desc>Some file</desc>
                                    <key id="d0" for="edge" attr.name="weight" attr.type="int"/>
                                    <key id="d1" for="node" attr.name="color" attr.type="string">
                                      <desc>Color</desc>
                                      <default>blue</default>
                                    </key>
                                    <key id="d2" for="edge" attr.name="distance" attr.type="double">
                                      <desc>Distance</desc>
                                      <default>1.5</default>
                                    </key>
                                    <graph edgedefault="undirected" parse.nodeids="canonical"
                                    parse.nodes="3" parse.edges="4">
                                    <node id="n0">
                                      <data key="d1">green</data>
                                    </node>
                                    <node id="n1"/>
                                    <node id="n2"/>
                                    <edge source="n0" target="n1">
                                      <data key="d0">1</data>
                                      <data key="d2">7.2</data>
                                    </edge>
                                    <edge source="n1" target="n2" directed='true'>
                                      <data key="d0">1</data>
                                    </edge>
                                    <edge source="n0" target="n2" directed='false'>
                                      <data key="d0">1</data>
                                    </edge></graph></graphml>""")
        graph = parse.load_graphml(self.stub_data_6)
        adjacency = graph.adjacency
        colors = graph.node_attribute.color
        distances = graph.edge_attribute.distance
        self.assertTrue((adjacency.indices == [1, 2, 0, 2, 0]).all())
        self.assertTrue((adjacency.indptr == [0, 2, 4, 5]).all())
        self.assertTrue((adjacency.data == [1, 1, 1, 1, 1]).all())
        self.assertTrue((colors == ['green', 'blue', 'blue']).all())
        self.assertTrue((distances == [7.2, 7.2, 1.5, 1.5, 1.5]).all())
        self.assertEqual(graph.meta.description, 'Some file')
        self.assertEqual(graph.meta.attributes.node.color, 'Color')
        self.assertEqual(graph.meta.attributes.edge.distance, 'Distance')
        remove(self.stub_data_6)

    def test_no_graphml(self):
        self.stub_data_7 = 'stub_7.graphml'
        with open(self.stub_data_7, "w") as graphml_file:
            graphml_file.write("""<?xml version='1.0' encoding='utf-8'?>
                                    <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
                                    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                                    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
                                    http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
                                    <key id="d0" for="edge" attr.name="weight" attr.type="int"/>
                                    </graphml>""")
        self.assertRaises(ValueError, parse.load_graphml, self.stub_data_7)
        remove(self.stub_data_7)

    def test_adjacency_list(self):
        self.stub_data_8 = 'stub_8.txt'
        with open(self.stub_data_8, "w") as text_file:
            text_file.write('%stub\n2\n3\n0\n1\n5\n4')
        graph = parse.load_adjacency_list(self.stub_data_8)
        adjacency = graph.adjacency
        self.assertTupleEqual(adjacency.shape, (6, 6))
        self.assertTrue((adjacency.indices == [2, 3, 0, 1, 5, 4]).all())
        self.assertTrue((adjacency.indptr == [0, 1, 2, 3, 4, 5, 6]).all())
        self.assertTrue((adjacency.data == [1, 1, 1, 1, 1, 1]).all())
        remove(self.stub_data_8)

    def test_bipartite_edge_list(self):
        self.stub_data_9 = 'stub_9.txt'
        with open(self.stub_data_9, "w") as text_file:
            text_file.write('%stub\n1 3\n4 5\n0 3')
        graph = parse.load_edge_list(self.stub_data_9, bipartite=True)
        biadjacency = graph.biadjacency
        self.assertTrue((biadjacency.indices == [0, 0, 1]).all())
        self.assertTrue((biadjacency.indptr == [0, 1, 2, 3]).all())
        self.assertTrue((biadjacency.data == [1, 1, 1]).all())
        remove(self.stub_data_9)

    def test_bipartite_adjacency_list(self):
        self.stub_data_10 = 'stub_10.txt'
        with open(self.stub_data_10, "w") as text_file:
            text_file.write('%stub\n3\n3\n0')
        graph = parse.load_adjacency_list(self.stub_data_10, bipartite=True)
        biadjacency = graph.biadjacency
        self.assertTupleEqual(biadjacency.shape, (3, 4))
        self.assertTrue((biadjacency.indices == [3, 3, 0]).all())
        self.assertTrue((biadjacency.indptr == [0, 1, 2, 3]).all())
        self.assertTrue((biadjacency.data == [1, 1, 1]).all())
        remove(self.stub_data_10)
