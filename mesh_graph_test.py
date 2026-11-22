#!/usr/bin/env python3

'''
mesh_graph_test.py

Unit tests for the surface graph: edge weights against Euclidean distance,
welding, shared edges, and disconnected surfaces.
'''

import math
import unittest
from mesh_graph import MeshAdjacencyGraph

class MeshAdjacencyGraphTest(unittest.TestCase):

    # A 3-4-5 right triangle, so every edge length is an integer
    def right_triangle(self):
        vertices = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 4.0, 0.0)]
        faces = [(0, 1, 2)]
        return MeshAdjacencyGraph(vertices, faces)

    def test_single_triangle_has_three_edges(self):
        graph = self.right_triangle()
        self.assertEqual(len(graph), 3)
        self.assertEqual(graph.edge_count(), 3)

    def test_edge_weights_are_euclidean_distances(self):
        graph = self.right_triangle()
        self.assertAlmostEqual(graph.adjacency[0][1], 3.0)
        self.assertAlmostEqual(graph.adjacency[0][2], 4.0)
        self.assertAlmostEqual(graph.adjacency[1][2], 5.0)

    def test_weights_match_math_dist_in_3d(self):
        vertices = [(1.0, 2.0, 3.0), (4.0, 6.0, 15.0), (0.0, 0.0, 0.0)]
        graph = MeshAdjacencyGraph(vertices, [(0, 1, 2)])
        for a in range(3):
            for b in graph.neighbors(a):
                self.assertAlmostEqual(graph.adjacency[a][b],
                                       math.dist(vertices[a], vertices[b]))

    def test_adjacency_is_symmetric(self):
        graph = self.right_triangle()
        for a, neighbors in graph.adjacency.items():
            for b, weight in neighbors.items():
                self.assertAlmostEqual(graph.adjacency[b][a], weight)

    def test_duplicate_vertices_are_welded(self):
        # The same square written as two triangles that do not share indices,
        # the way an .obj splits vertices at a seam
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        graph = MeshAdjacencyGraph(vertices, [(0, 1, 2), (3, 4, 5)])
        self.assertEqual(len(graph.vertices), 4)
        self.assertEqual(len(graph), 4)
        # Four sides plus the shared diagonal, counted once
        self.assertEqual(graph.edge_count(), 5)
        self.assertEqual(len(graph.components()), 1)

    def test_shared_edge_counted_once(self):
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        graph = MeshAdjacencyGraph(vertices, [(0, 1, 2), (0, 2, 3)])
        self.assertEqual(graph.edge_count(), 5)

    def test_disconnected_triangles_are_two_components(self):
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                    (9.0, 9.0, 9.0), (10.0, 9.0, 9.0), (9.0, 10.0, 9.0)]
        graph = MeshAdjacencyGraph(vertices, [(0, 1, 2), (3, 4, 5)])
        groups = graph.components()
        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(len(group) for group in groups), [3, 3])
        self.assertEqual(graph.walk(0), {0, 1, 2})

    def test_components_sorted_largest_first(self):
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                    (9.0, 9.0, 9.0), (10.0, 9.0, 9.0), (9.0, 10.0, 9.0)]
        graph = MeshAdjacencyGraph(vertices, [(0, 1, 2), (0, 2, 3), (4, 5, 6)])
        sizes = [len(group) for group in graph.components()]
        self.assertEqual(sizes, [4, 3])
        self.assertEqual(len(graph.largest_component()), 4)

    def test_collapsed_triangle_adds_no_self_edge(self):
        # Two corners within tolerance weld to one, leaving a single edge
        vertices = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        graph = MeshAdjacencyGraph(vertices, [(0, 1, 2)])
        self.assertEqual(len(graph.vertices), 2)
        self.assertEqual(graph.edge_count(), 1)
        for vertex in graph.adjacency:
            self.assertNotIn(vertex, graph.neighbors(vertex))

    def test_empty_mesh(self):
        graph = MeshAdjacencyGraph([], [])
        self.assertEqual(len(graph), 0)
        self.assertEqual(graph.edge_count(), 0)
        self.assertEqual(graph.components(), [])
        self.assertEqual(graph.largest_component(), set())

if __name__ == '__main__':
    unittest.main()
