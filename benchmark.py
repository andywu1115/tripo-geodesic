#!/usr/bin/env python3

'''
benchmark.py

Time the geodesic solver across the four meshes against three baselines,
check that they all agree, and plot the result.

What each timing isolates:

  our solver with the hand written MinHeap from HW09
  our solver with the HW07 sorted linked list priority queue, which is the
    same algorithm with an O(N) frontier instead of an O(log N) one
  our solver with heapq swapped in, the same binary heap written in C
  networkx.single_source_dijkstra_path_length, which differs in everything
    and shows what a tuned library costs on the same graph

The first three differ only in the priority queue, because the solver takes
its frontier as a parameter. That is what makes the comparison mean
something: against networkx alone you cannot tell whether a gap comes from
the heap, the graph representation or the inner loop.

Every run is single source to exhaustion from the same origin on the same
graph, and the distance dicts are compared before any time is reported, so
a fast wrong answer cannot pass.
'''

import csv
import math
import sys
import time
from heapq import heappop, heappush

import networkx

from geodesic import compute_geodesic_field
from mesh_graph import MeshAdjacencyGraph, filter_faces_above
from min_heap import MinHeap
from obj_parser import load_obj
from priority_queue import PriorityQueue

MESHES = ['meshes/mesh_00500.obj', 'meshes/mesh_02000.obj',
          'meshes/mesh_10000.obj', 'meshes/mesh_20000.obj']

REPEATS = 3

# The linked list frontier is tens of times slower, so it runs on the
# smaller meshes only and once rather than three times. A single run is
# enough when the measurement is this far above the noise.
LINKED_LIST_MAX_VERTICES = 6000
LINKED_LIST_REPEATS = 1

class HeapqFrontier:
    '''
    The MinHeap interface backed by Python's heapq, which is the same binary
    heap algorithm written in C. The solver only calls insert, remove and
    len, so either class drops into it unchanged.
    '''
    def __init__(self) -> None:
        self.heap_array = []

    def insert(self, value) -> None:
        heappush(self.heap_array, value)

    def remove(self):
        return heappop(self.heap_array)

    def __len__(self) -> int:
        return len(self.heap_array)

class LinkedListFrontier:
    '''
    The MinHeap interface backed by the HW07 priority queue, which keeps a
    doubly linked list in ascending order.

    HW07's own pop() returns the largest value from the tail, and Dijkstra
    needs the smallest, so this pops index 0 instead. That reaches the head
    dummy node in one step, so removal is O(1), cheaper than the heap's
    O(log N). The cost is all in push: insert_ordered walks from the head
    until it finds a larger value, which is O(N) in the size of the
    frontier. That trade is the whole point of the comparison.
    '''
    def __init__(self) -> None:
        self.queue = PriorityQueue()

    def insert(self, value) -> None:
        self.queue.push(value)

    def remove(self):
        return self.queue.dll.pop(0)

    def __len__(self) -> int:
        return len(self.queue)

def build_networkx_graph(graph: MeshAdjacencyGraph) -> networkx.Graph:
    '''The same surface graph as a networkx Graph. Built once, never timed.'''
    nx_graph = networkx.Graph()
    for vertex, neighbors in graph.adjacency.items():
        for neighbor, weight in neighbors.items():
            nx_graph.add_edge(vertex, neighbor, weight=weight)
    return nx_graph

def best_of(repeats: int, function, *args):
    '''
    Run something several times and keep the fastest. The fastest run is the
    one least disturbed by other work on the machine, so it estimates the
    algorithm's own cost better than an average, which only measures how
    busy the laptop was.
    '''
    best = math.inf
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(*args)
        best = min(best, time.perf_counter() - start)
    return best, result

def check_same(expected: dict, actual: dict, label: str,
               tolerance: float = 1e-9) -> None:
    '''Stop the benchmark if a baseline disagrees with our solver.'''
    if set(expected) != set(actual):
        raise AssertionError(f'{label} reached a different set of vertices')
    for vertex, distance in expected.items():
        if abs(distance - actual[vertex]) > tolerance:
            raise AssertionError(f'{label} disagrees at vertex {vertex}: '
                                 f'{distance} vs {actual[vertex]}')

def fitted_exponent(sizes, times) -> float:
    '''
    Least squares slope of log time against log V, which is the exponent b
    in t proportional to V to the b. On log axes that is the slope of the
    plotted line. A heap frontier should land a little above 1, since
    V log V grows slightly faster than V.
    '''
    pairs = [(math.log(size), math.log(seconds))
             for size, seconds in zip(sizes, times) if seconds]
    if len(pairs) < 2:
        return float('nan')

    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = sum((x - mean_x) ** 2 for x, _ in pairs)
    return numerator / denominator

def run_mesh(path: str, min_height=None, repeats: int = REPEATS) -> dict:
    vertices, faces = load_obj(path)
    if min_height is not None:
        faces = filter_faces_above(vertices, faces, min_height)

    graph = MeshAdjacencyGraph(vertices, faces)
    component = graph.largest_component()
    if not component:
        raise ValueError(f'{path} has no surface to route on')

    # Route from inside the largest component. Dijkstra never leaves the
    # component it starts in, so that component's size is the V the runtime
    # actually depends on, not the vertex count of the whole file.
    origin = min(component)
    edges = sum(len(graph.neighbors(vertex)) for vertex in component) // 2

    mine_s, (mine, _) = best_of(repeats, compute_geodesic_field,
                                graph, origin, MinHeap)
    heapq_s, (with_heapq, _) = best_of(repeats, compute_geodesic_field,
                                       graph, origin, HeapqFrontier)

    nx_graph = build_networkx_graph(graph)
    networkx_s, nx_distances = best_of(
        repeats, networkx.single_source_dijkstra_path_length, nx_graph, origin)

    check_same(mine, with_heapq, 'heapq frontier')
    check_same(mine, nx_distances, 'networkx')

    linked_s = None
    if len(mine) <= LINKED_LIST_MAX_VERTICES:
        linked_s, (with_linked, _) = best_of(
            LINKED_LIST_REPEATS, compute_geodesic_field,
            graph, origin, LinkedListFrontier)
        check_same(mine, with_linked, 'linked list frontier')

    return {'mesh': path.split('/')[-1],
            'vertices': len(mine),
            'edges': edges,
            'mine_s': mine_s,
            'linked_s': linked_s,
            'heapq_s': heapq_s,
            'networkx_s': networkx_s}

def plot(rows: list[dict], path: str = 'benchmark.png') -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sizes = [row['vertices'] for row in rows]
    figure, axes = plt.subplots(figsize=(7.5, 5.5))

    series = (('linked_s', 'our solver, HW07 sorted linked list', 'D'),
              ('mine_s', 'our solver, hand written MinHeap', 'o'),
              ('heapq_s', 'our solver, heapq frontier', 's'),
              ('networkx_s', 'networkx single source dijkstra', '^'))

    for key, label, marker in series:
        measured = [(row['vertices'], row[key]) for row in rows if row[key]]
        axes.plot([size for size, _ in measured],
                  [seconds for _, seconds in measured],
                  marker=marker, label=label)

    # A V log V line scaled through our first measurement. On log axes a
    # matching slope is the visual check that the growth is O(V log V); the
    # vertical gap between curves is the constant factor.
    reference = [size * math.log(size) for size in sizes]
    scale = rows[0]['mine_s'] / reference[0]
    axes.plot(sizes, [scale * value for value in reference], '--',
              color='gray', label='V log V reference')

    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlabel('vertices reached from the origin (V)')
    axes.set_ylabel('seconds')
    axes.set_title('Dijkstra on a 3D mesh surface graph')
    axes.legend(fontsize=8)
    axes.grid(True, which='both', alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    print(f'wrote {path}')

# Main Execution

def main(args=None) -> None:
    if args is None:
        args = sys.argv[1:]
    min_height = float(args[0]) if args else None

    rows = []
    for path in MESHES:
        row = run_mesh(path, min_height)
        rows.append(row)
        linked = f'{row["linked_s"]:8.4f}s' if row['linked_s'] else '  skipped'
        print(f'{row["mesh"]:18} V={row["vertices"]:>6,} E={row["edges"]:>7,} '
              f'linked list {linked}  mine {row["mine_s"]:7.4f}s  '
              f'heapq {row["heapq_s"]:7.4f}s  '
              f'networkx {row["networkx_s"]:7.4f}s')

    print()
    print(f'{"mesh":18} {"V":>7} {"linked/mine":>12} {"mine/heapq":>11} '
          f'{"mine/networkx":>14} {"s per V log V":>15}')
    for row in rows:
        ratio = (f'{row["linked_s"] / row["mine_s"]:>11.1f}x'
                 if row['linked_s'] else f'{"":>12}')
        per_operation = row['mine_s'] / (row['vertices']
                                         * math.log(row['vertices']))
        print(f'{row["mesh"]:18} {row["vertices"]:>7,} {ratio} '
              f'{row["mine_s"] / row["heapq_s"]:>10.2f}x '
              f'{row["mine_s"] / row["networkx_s"]:>13.2f}x '
              f'{per_operation:>15.2e}')

    print()
    print('fitted exponent b, where seconds grow as V to the b:')
    sizes = [row['vertices'] for row in rows]
    for key, label in (('linked_s', 'HW07 linked list'),
                       ('mine_s', 'our MinHeap'),
                       ('heapq_s', 'heapq'),
                       ('networkx_s', 'networkx')):
        print(f'  {label:18} {fitted_exponent(sizes, [row[key] for row in rows]):.2f}')

    with open('benchmark.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('wrote benchmark.csv')

    plot(rows)

if __name__ == '__main__':
    main()
