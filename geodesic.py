#!/usr/bin/env python3

'''
geodesic.py

Shortest surface paths across a 3D mesh: Dijkstra's algorithm on the mesh
surface graph, with the hand written MinHeap as the frontier.

This extends find_cheapest_flight_plan from HW10B. The shape is unchanged: a
frontier of (distance, vertex, source) tuples, a dict recording where each
settled vertex was reached from, and lazy deletion, where a vertex popped a
second time is skipped. Four things differ.

The frontier is the HW09 MinHeap rather than heapq.
Weights are float Euclidean distances rather than integer ticket prices.
The search runs to exhaustion rather than returning at the destination, so
one run yields the distance to every reachable vertex, which is what the
benchmark times and what a spatial analysis wants.
An unreachable destination reports infinity rather than a cost of zero.
'''

import math
import sys

from mesh_graph import MeshAdjacencyGraph, filter_faces_above, UP
from min_heap import MinHeap
from obj_parser import load_obj

# Types

# Each settled vertex maps to its total distance from the origin
Distances = dict[int, float]

# Each settled vertex maps to the vertex before it on its shortest path,
# as in HW10B's Plan. The origin is its own predecessor.
Predecessors = dict[int, int]

# Functions

def compute_geodesic_field(graph: MeshAdjacencyGraph, origin: int,
                           frontier_factory=MinHeap
                           ) -> tuple[Distances, Predecessors]:
    '''
    Dijkstra from one origin across the whole surface.

    Lazy deletion rather than a decrease key: when a shorter route to a
    vertex is found, a second entry is pushed instead of the existing one
    being updated, and whichever entry comes off the heap first is the
    shortest, so later entries for that vertex are skipped. The heap can hold
    up to E entries instead of V, which costs memory but keeps every heap
    operation O(log V) with no index bookkeeping. A real decrease key would
    need a dict from each vertex to its position in the array.
    '''
    frontier = frontier_factory()
    frontier.insert((0.0, origin, origin))

    distances: Distances = {}
    predecessors: Predecessors = {}

    while len(frontier) > 0:
        distance, vertex, source = frontier.remove()

        # Already settled by a shorter route, so this entry is stale
        if vertex in distances:
            continue

        distances[vertex] = distance
        predecessors[vertex] = source

        for neighbor, weight in graph.neighbors(vertex).items():
            if neighbor not in distances:
                frontier.insert((distance + weight, neighbor, vertex))

    return distances, predecessors

def reconstruct_path(predecessors: Predecessors,
                     origin: int, destination: int) -> list[int]:
    '''
    The vertices from origin to destination in order, or an empty list when
    the destination was never reached.

    HW10B reconstructs with a recursive generator. This walks the chain in a
    loop instead, because each recursive call is a stack frame and a path on
    a denser mesh could reach Python's recursion limit. The walk ends at the
    origin because the origin is recorded as its own predecessor.
    '''
    if destination not in predecessors:
        return []

    path = []
    vertex = destination
    while vertex != origin:
        path.append(vertex)
        vertex = predecessors[vertex]
    path.append(origin)
    path.reverse()
    return path

def shortest_surface_path(graph: MeshAdjacencyGraph, origin: int,
                          destination: int) -> tuple[float, list[int]]:
    '''
    Surface distance and the path that realizes it.
    Returns (inf, []) when no path exists along the surface.
    '''
    distances, predecessors = compute_geodesic_field(graph, origin)

    if destination not in distances:
        return math.inf, []

    return distances[destination], reconstruct_path(predecessors, origin,
                                                    destination)

def path_length(graph: MeshAdjacencyGraph, path: list[int]) -> float:
    '''
    Add up the edge weights along a path. This recomputes the distance from
    the graph alone, so it checks the solver's own total without trusting it.
    '''
    return sum(graph.adjacency[path[i]][path[i + 1]]
               for i in range(len(path) - 1))

def extreme_vertices(graph: MeshAdjacencyGraph, component: set[int],
                     axis: int = 0) -> tuple[int, int]:
    '''
    The two vertices of a component at opposite ends of one axis, which gives
    a coast to coast demo path without comparing every pair.
    '''
    low = min(component, key=lambda vertex: graph.vertices[vertex][axis])
    high = max(component, key=lambda vertex: graph.vertices[vertex][axis])
    return low, high

def format_point(point: tuple[float, float, float]) -> str:
    return '(' + ', '.join(f'{coordinate:.3f}' for coordinate in point) + ')'

# Main Execution

def main(args=None) -> None:
    if args is None:
        args = sys.argv[1:]
    if not 1 <= len(args) <= 2:
        print('Usage: geodesic.py mesh.obj [min_height]')
        sys.exit(1)

    vertices, faces = load_obj(args[0])

    if len(args) == 2:
        min_height = float(args[1])
        faces = filter_faces_above(vertices, faces, min_height)
        print(f'routing only on surface above {min_height}')

    graph = MeshAdjacencyGraph(vertices, faces)
    component = graph.largest_component()
    if not component:
        print('no surface to route on')
        return

    print(f'{len(graph):,} vertices with edges, '
          f'largest component {len(component):,}')

    origin, destination = extreme_vertices(graph, component)
    distance, path = shortest_surface_path(graph, origin, destination)

    print(f'origin      {origin} at {format_point(graph.vertices[origin])}')
    print(f'destination {destination} at '
          f'{format_point(graph.vertices[destination])}')

    if not path:
        print('no surface path between them')
        return

    straight = math.dist(graph.vertices[origin], graph.vertices[destination])
    lowest = min(graph.vertices[vertex][UP] for vertex in path)
    print(f'surface distance:  {distance:.4f} over {len(path)} vertices')
    print(f'resummed from edges: {path_length(graph, path):.4f}')
    print(f'straight line:     {straight:.4f} '
          f'(surface is {distance / straight:.2f}x longer)')
    print(f'lowest point on the path: {lowest:.4f}')

if __name__ == '__main__':
    main()
