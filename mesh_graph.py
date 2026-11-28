#!/usr/bin/env python3

'''
mesh_graph.py

Build an undirected, weighted adjacency list from a triangle mesh, where each
mesh vertex is a node and each triangle edge is an edge weighted by the 3D
Euclidean distance between its endpoints. This is the graph Dijkstra walks,
so a path through it is a path along the mesh surface.

Two things the raw file forces on us. An .obj writes a vertex once per
texture or normal seam, so geometrically identical corners arrive as
different indices and would split the surface into pieces that do not touch;
welding by rounded coordinates puts them back together. And an AI generated
mesh often carries loose geometry (here, the trees), which forms its own
connected components, so the graph reports its components and the caller
decides which surface to route on.
'''

import math
import sys
from collections import deque

from obj_parser import load_obj

# Types

# Each vertex maps to its neighbors and the distance along the edge to each,
# as in the weighted graphs from EX33: {vertex: {neighbor: weight, }}
AdjacencyList = dict[int, dict[int, float]]

# Index of the up axis in a vertex tuple. These meshes are Y up: Y runs from
# 0 at the flat base to about 0.3 at the crater rim, while X and Z each span
# the full width.
UP = 1

# Functions

# Keep only the faces whose three corners all sit above min_height.
# Dropping the underside leaves the navigable surface: without this, a path
# between two coastal points runs straight across the flat base instead of
# following the terrain, which is the correct geodesic on the mesh as written
# and the wrong answer for navigation. Requiring all three corners, rather
# than any one, means no kept triangle straddles the cut.
def filter_faces_above(vertices: list[tuple[float, float, float]],
                       faces: list[tuple[int, int, int]],
                       min_height: float,
                       axis: int = UP) -> list[tuple[int, int, int]]:
    return [face for face in faces
            if all(vertices[corner][axis] > min_height for corner in face)]

class MeshAdjacencyGraph:
    # Build the graph from a vertex list and a triangle list. Vertices within
    # tolerance of each other are treated as one node.
    def __init__(self, vertices: list[tuple[float, float, float]],
                 faces: list[tuple[int, int, int]],
                 tolerance: float = 1e-6) -> None:
        self.tolerance = tolerance
        self.vertices, self.index_map = self._weld(vertices, tolerance)
        index_map = self.index_map
        self.adjacency: AdjacencyList = {}

        for face in faces:
            a, b, c = (index_map[corner] for corner in face)
            self._add_edge(a, b)
            self._add_edge(b, c)
            self._add_edge(c, a)

    # Merge vertices that share a position, returning the unique positions and
    # a map from each original index to its index in that list.
    # Rounding to a grid of size tolerance turns "close enough" into dictionary
    # key equality, which is O(1) per vertex, so welding is O(V) overall. A
    # true nearest-neighbor merge would need a spatial index and is not worth
    # it here: seam duplicates hold exactly the same coordinates.
    @staticmethod
    def _weld(vertices, tolerance):
        unique = []
        key_to_index = {}
        index_map = []

        for vertex in vertices:
            key = tuple(round(coordinate / tolerance) for coordinate in vertex)
            if key not in key_to_index:
                key_to_index[key] = len(unique)
                unique.append(vertex)
            index_map.append(key_to_index[key])

        return unique, index_map

    # Add one undirected edge, stored in both directions as in HW10A.
    # Writing the same edge twice is harmless: the weight is the same, so the
    # second write replaces the first with an identical value. That is what
    # makes a shared triangle edge count once rather than twice.
    def _add_edge(self, a: int, b: int) -> None:
        # A triangle whose corners welded together has no length to walk
        if a == b:
            return

        weight = math.dist(self.vertices[a], self.vertices[b])
        self.adjacency.setdefault(a, {})[b] = weight
        self.adjacency.setdefault(b, {})[a] = weight

    # Neighbors of a vertex, empty if it has no edges.
    # A plain dict with .get, not a defaultdict: reading a missing vertex from
    # a defaultdict inserts it, and inserting while walking the graph changes
    # the dictionary's size mid-iteration.
    def neighbors(self, vertex: int) -> dict[int, float]:
        return self.adjacency.get(vertex, {})

    # Number of vertices that have at least one edge, supporting len(graph)
    def __len__(self) -> int:
        return len(self.adjacency)

    # Number of undirected edges. Each is stored twice, once per direction.
    def edge_count(self) -> int:
        return sum(len(neighbors) for neighbors in self.adjacency.values()) // 2

    # Every vertex reachable from the origin, by breadth-first search with the
    # frontier and visited pattern from HW10A
    def walk(self, origin: int) -> set[int]:
        frontier = deque([origin])
        visited: set[int] = set()

        while frontier:
            vertex = frontier.popleft()

            if vertex in visited:
                continue

            visited.add(vertex)

            for neighbor in self.neighbors(vertex):
                if neighbor not in visited:
                    frontier.append(neighbor)

        return visited

    # The connected components, largest first. Each is the set of vertices in
    # one piece of surface; no path exists between two different components.
    def components(self) -> list[set[int]]:
        seen: set[int] = set()
        groups: list[set[int]] = []

        for vertex in self.adjacency:
            if vertex not in seen:
                group = self.walk(vertex)
                seen |= group
                groups.append(group)

        groups.sort(key=len, reverse=True)
        return groups

    # The largest connected component, which for these meshes is the island
    # itself rather than the loose geometry sitting on it
    def largest_component(self) -> set[int]:
        groups = self.components()
        return groups[0] if groups else set()

# Main Execution

def report(graph: MeshAdjacencyGraph, label: str) -> None:
    groups = graph.components()
    print(f'{label}:')
    print(f'  {len(graph):,} vertices with edges, '
          f'{graph.edge_count():,} undirected edges')
    print(f'  {len(groups):,} connected components')
    if groups:
        largest = len(groups[0])
        print(f'  largest component: {largest:,} vertices '
              f'({100 * largest / len(graph):.1f}%)')

def main(args=None) -> None:
    if args is None:
        args = sys.argv[1:]
    if not 1 <= len(args) <= 2:
        print('Usage: mesh_graph.py mesh.obj [min_height]')
        sys.exit(1)

    vertices, faces = load_obj(args[0])
    if not vertices:
        print('empty mesh')
        return

    print(f'{len(vertices):,} vertices in the file, {len(faces):,} triangles')

    # Where the underside sits. Vertices bunched at the lowest heights are the
    # flat base and the thin skirt at the waterline, and those are what give a
    # path a shortcut that does not follow the terrain.
    low = min(vertex[UP] for vertex in vertices)
    high = max(vertex[UP] for vertex in vertices)
    span = high - low
    print(f'height runs {low:.4f} to {high:.4f}')
    for fraction in (0.0, 0.001, 0.01, 0.02, 0.05, 0.10):
        threshold = low + fraction * span
        count = sum(1 for vertex in vertices if vertex[UP] <= threshold)
        print(f'  at or below {threshold:.4f} ({fraction:.1%} of height): '
              f'{count:,} vertices ({100 * count / len(vertices):.1f}%)')

    report(MeshAdjacencyGraph(vertices, faces), 'whole mesh')

    if len(args) == 2:
        min_height = float(args[1])
        kept = filter_faces_above(vertices, faces, min_height)
        print(f'{len(kept):,} of {len(faces):,} triangles sit entirely '
              f'above {min_height}')
        report(MeshAdjacencyGraph(vertices, kept), f'above {min_height}')

if __name__ == '__main__':
    main()
