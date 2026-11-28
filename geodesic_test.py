#!/usr/bin/env python3

'''
geodesic_test.py

Unit tests for the mesh Dijkstra: hand computed distances on small meshes,
detours, unreachable destinations, and agreement between the solver's total
and the edge weights along the path it returns.
'''

import math
import unittest

from geodesic import (compute_geodesic_field, path_length, reconstruct_path,
                      shortest_surface_path)
from mesh_graph import MeshAdjacencyGraph

class GeodesicTest(unittest.TestCase):

    # A 3-4-5 right triangle, so every edge length is an integer
    def right_triangle(self):
        vertices = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 4.0, 0.0)]
        return MeshAdjacencyGraph(vertices, [(0, 1, 2)])

    # Two triangles sharing edge 1-2, so vertices 0 and 3 are not adjacent
    # and any route between them passes through 1 or 2
    def strip(self):
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                    (0.0, 0.0, 2.0), (1.0, 0.0, 1.0)]
        return MeshAdjacencyGraph(vertices, [(0, 1, 2), (1, 3, 2)])

    def test_direct_edge_wins_when_shortest(self):
        # 1 to 2 is 5 directly, or 3 + 4 = 7 through vertex 0
        graph = self.right_triangle()
        distance, path = shortest_surface_path(graph, 1, 2)
        self.assertAlmostEqual(distance, 5.0)
        self.assertEqual(path, [1, 2])

    def test_detour_through_a_middle_vertex(self):
        # 0 to 3 has no direct edge: 0-1-3 is 2.0, 0-2-3 is longer
        graph = self.strip()
        distance, path = shortest_surface_path(graph, 0, 3)
        self.assertAlmostEqual(distance, 2.0)
        self.assertEqual(path, [0, 1, 3])

    def test_origin_to_itself_is_zero(self):
        graph = self.right_triangle()
        distance, path = shortest_surface_path(graph, 0, 0)
        self.assertAlmostEqual(distance, 0.0)
        self.assertEqual(path, [0])

    def test_unreachable_destination_is_infinite(self):
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                    (9.0, 9.0, 9.0), (10.0, 9.0, 9.0), (9.0, 10.0, 9.0)]
        graph = MeshAdjacencyGraph(vertices, [(0, 1, 2), (3, 4, 5)])
        distance, path = shortest_surface_path(graph, 0, 4)
        self.assertEqual(distance, math.inf)
        self.assertEqual(path, [])

    def test_field_covers_exactly_the_origin_component(self):
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                    (9.0, 9.0, 9.0), (10.0, 9.0, 9.0), (9.0, 10.0, 9.0)]
        graph = MeshAdjacencyGraph(vertices, [(0, 1, 2), (3, 4, 5)])
        distances, predecessors = compute_geodesic_field(graph, 0)
        self.assertEqual(set(distances), {0, 1, 2})
        self.assertEqual(set(predecessors), {0, 1, 2})
        self.assertEqual(predecessors[0], 0)

    def test_distances_never_decrease_along_a_path(self):
        graph = self.strip()
        distances, predecessors = compute_geodesic_field(graph, 0)
        for vertex, distance in distances.items():
            source = predecessors[vertex]
            self.assertLessEqual(distances[source], distance + 1e-12)

    def test_total_matches_the_edges_on_the_path(self):
        graph = self.strip()
        distance, path = shortest_surface_path(graph, 0, 3)
        self.assertAlmostEqual(distance, path_length(graph, path))

    def test_reconstruct_path_without_the_destination(self):
        self.assertEqual(reconstruct_path({0: 0}, 0, 7), [])

if __name__ == '__main__':
    unittest.main()
