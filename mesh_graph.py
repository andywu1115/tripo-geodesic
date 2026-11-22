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

class MeshAdjacencyGraph:
    # Build the graph from a vertex list and a triangle list. Vertices within
    # tolerance of each other are treated as one node.
    def __init__(self, vertices: list[tuple[float, float, float]],
                 faces: list[tuple[int, int, int]],
                 tolerance: float = 1e-6) -> None:
        self.tolerance = tolerance
        self.vertices, index_map = self._weld(vertices, tolerance)
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

def main(args=None) -> None:
    if args is None:
        args = sys.argv[1:]
    if len(args) != 1:
        print('Usage: mesh_graph.py mesh.obj')
        sys.exit(1)

    vertices, faces = load_obj(args[0])
    graph = MeshAdjacencyGraph(vertices, faces)
    groups = graph.components()

    print(f'{len(vertices):,} vertices in the file')
    print(f'{len(graph.vertices):,} after welding, '
          f'{len(graph):,} of them in at least one edge')
    print(f'{len(faces):,} triangles, {graph.edge_count():,} undirected edges')
    print(f'{len(groups):,} connected components')

    if groups:
        largest = len(groups[0])
        print(f'largest component: {largest:,} vertices '
              f'({100 * largest / len(graph):.1f}% of connected vertices)')
        print(f'component sizes, largest first: '
              f'{[len(group) for group in groups[:8]]}')

if __name__ == '__main__':
    main()
