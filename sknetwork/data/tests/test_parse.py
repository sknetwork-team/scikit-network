#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for parse.py"""

import unittest
from os import remove

import numpy as np

from sknetwork.data import parse


class TestParser(unittest.TestCase):

    def test_unlabeled_unweighted(self):
        self.stub_data_1 = 'stub_1.txt'
        with open(self.stub_data_1, "w") as text_file:
            text_file.write('%stub\n1 3\n4 5\n0 2')
        adjacency = parse.from_csv(self.stub_data_1)
        self.assertTrue((adjacency.indices == [2, 3, 0, 1, 5, 4]).all())
        self.assertTrue((adjacency.indptr == [0, 1, 2, 3, 4, 5, 6]).all())
        self.assertTrue((adjacency.data == [1, 1, 1, 1, 1, 1]).all())
        adjacency = parse.from_csv(self.stub_data_1, shape=(7, 7))
        self.assertTrue((adjacency.shape == (7, 7)))
        biadjacency = parse.from_csv(self.stub_data_1, bipartite=True, shape=(7, 9))
        self.assertTrue((biadjacency.shape == (7, 9)))
        remove(self.stub_data_1)

    def test_labeled_weighted(self):
        self.stub_data_2 = 'stub_2.txt'
        with open(self.stub_data_2, "w") as text_file:
            text_file.write('%stub\nf, e, 5\na, d, 6\nc, b, 1')
        graph = parse.from_csv(self.stub_data_2)
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
        graph = parse.from_csv(self.stub_data_4, reindex=True)
        adjacency = graph.adjacency
        names = graph.names
        self.assertTrue((adjacency.data == [1, 1, 1, 1, 1, 1]).all())
        self.assertTrue((names == [0, 12, 14, 31, 42, 50]).all())
        remove(self.stub_data_4)

    def test_wrong_format(self):
        self.stub_data_3 = 'stub_3.txt'
        with open(self.stub_data_3, "w") as text_file:
            text_file.write('%stub\n1 3 a\n4 5 b\n0 2 e')
        self.assertRaises(ValueError, parse.from_csv, self.stub_data_3)
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
        graph = parse.from_graphml(self.stub_data_5)
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
        graph = parse.from_graphml(self.stub_data_6)
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
        self.assertRaises(ValueError, parse.from_graphml, self.stub_data_7)
        remove(self.stub_data_7)

    def test_csv_adjacency(self):
        self.stub_data_8 = 'stub_8.txt'
        with open(self.stub_data_8, "w") as text_file:
            text_file.write('%stub\n2\n3\n0\n1\n5\n4')
        adjacency = parse.from_csv(self.stub_data_8)
        self.assertTupleEqual(adjacency.shape, (6, 6))
        self.assertTrue((adjacency.indices == [2, 3, 0, 1, 5, 4]).all())
        self.assertTrue((adjacency.indptr == [0, 1, 2, 3, 4, 5, 6]).all())
        self.assertTrue((adjacency.data == [2, 2, 2, 2, 2, 2]).all())
        graph = parse.from_csv(self.stub_data_8, matrix_only=False)
        adjacency = graph.adjacency
        self.assertTupleEqual(adjacency.shape, (6, 6))
        remove(self.stub_data_8)
        # without comment line
        self.stub_data_8 = 'stub_8.txt'
        with open(self.stub_data_8, "w") as text_file:
            text_file.write('2\n3\n0\n1\n5\n4')
        adjacency = parse.from_csv(self.stub_data_8)
        self.assertTupleEqual(adjacency.shape, (6, 6))
        remove(self.stub_data_8)

    def test_csv_bipartite(self):
        self.stub_data_9 = 'stub_9.txt'
        with open(self.stub_data_9, "w") as text_file:
            text_file.write('#stub\n1 3\n4 5\n0 3')
        graph = parse.from_csv(self.stub_data_9, bipartite=True, reindex=True)
        biadjacency = graph.biadjacency
        self.assertTrue((biadjacency.indices == [0, 0, 1]).all())
        self.assertTrue((biadjacency.indptr == [0, 1, 2, 3]).all())
        self.assertTrue((biadjacency.data == [1, 1, 1]).all())
        biadjacency = parse.from_csv(self.stub_data_9, bipartite=True)
        self.assertTrue(biadjacency.shape == (5, 6))
        remove(self.stub_data_9)

    def test_edge_list(self):
        edge_list_1 = [('Alice', 'Bob'), ('Carol', 'Alice')]
        graph = parse.from_edge_list(edge_list_1)
        adjacency = graph.adjacency
        names = graph.names
        self.assertTrue((names == ['Alice', 'Bob', 'Carol']).all())
        self.assertTupleEqual(adjacency.shape, (3, 3))
        self.assertTrue((adjacency.indptr == [0, 2, 3, 4]).all())
        self.assertTrue((adjacency.indices == [1, 2, 0, 0]).all())
        self.assertTrue((adjacency.data == [1, 1, 1, 1]).all())

        edge_list_2 = [('Alice', 'Bob', 4), ('Carol', 'Alice', 6)]
        graph = parse.from_edge_list(edge_list_2)
        adjacency = graph.adjacency
        names = graph.names
        self.assertTrue((names == ['Alice', 'Bob', 'Carol']).all())
        self.assertTupleEqual(adjacency.shape, (3, 3))
        self.assertTrue((adjacency.indptr == [0, 2, 3, 4]).all())
        self.assertTrue((adjacency.indices == [1, 2, 0, 0]).all())
        self.assertTrue((adjacency.data == [4, 6, 4, 6]).all())

        edge_list_3 = [('Alice', 'Bob'), ('Carol', 'Alice'), ('Alice', 'Bob')]
        graph = parse.from_edge_list(edge_list_3, directed=True)
        adjacency = graph.adjacency
        names = graph.names
        self.assertTrue((names == ['Alice', 'Bob', 'Carol']).all())
        self.assertTupleEqual(adjacency.shape, (3, 3))
        self.assertTrue((adjacency.data == [2, 1]).all())

        adjacency = parse.from_edge_list(edge_list_3, directed=True, matrix_only=True)
        self.assertTupleEqual(adjacency.shape, (3, 3))

        graph = parse.from_edge_list(edge_list_3, directed=True, weighted=False)
        adjacency = graph.adjacency
        self.assertTrue((adjacency.data == [1, 1]).all())

        edge_list = np.array([[0, 1, 1], [1, 2, 2]])
        adjacency = parse.from_edge_list(edge_list, weighted=True, matrix_only=True)
        self.assertTrue((adjacency.data == np.array([1, 1, 2, 2])).all())

        edge_list = np.array([[0, 1, 2], [0, 1, 1], [1, 2, 1]])
        adjacency = parse.from_edge_list(edge_list, weighted=True, matrix_only=True, sum_duplicates=False)
        self.assertTrue(adjacency.data[0] == 2)

    def test_adjacency_list(self):
        edge_list_4 = {'Alice': ['Bob', 'Carol'], 'Bob': ['Carol']}
        graph = parse.from_adjacency_list(edge_list_4, directed=True)
        adjacency = graph.adjacency
        names = graph.names
        self.assertTrue((names == ['Alice', 'Bob', 'Carol']).all())
        self.assertTupleEqual(adjacency.shape, (3, 3))
        self.assertTrue((adjacency.data == [1, 1, 1]).all())

        edge_list_5 = [[0, 1, 2], [2, 3]]
        adjacency = parse.from_adjacency_list(edge_list_5, directed=True)
        self.assertTupleEqual(adjacency.shape, (4, 4))
        self.assertTrue((adjacency.data == [1, 1, 1, 1, 1]).all())

        self.assertRaises(TypeError, parse.from_adjacency_list, {2, 3})

    def test_bad_format_edge_list(self):
        edge_list_2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.assertRaises(ValueError, parse.from_edge_list, edge_list_2)
        edge_list_3 = 'ab cd'
        self.assertRaises(TypeError, parse.from_edge_list, edge_list_3)

    def test_is_number(self):
        self.assertTrue(parse.is_number(3))
        self.assertFalse(parse.is_number('a'))

