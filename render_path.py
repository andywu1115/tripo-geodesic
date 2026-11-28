#!/usr/bin/env python3

'''
render_path.py

Draw a geodesic path across the mesh, as a PNG, in three panels.

Each panel does one job. The 3D surface shows what the mesh is, and it is
the only view that reads as an island at a glance. The plan view shows the
route, because matplotlib gives a line and a surface one depth value each
and draws one wholly over the other, so the 3D view cannot be trusted to
show a path that runs between rock formations taller than it is. The
elevation profile carries the height the plan view throws away.

The proposal names trimesh for the overlay. This uses matplotlib instead:
it is already a dependency for the runtime plots, it renders with no OpenGL
context, and it draws from the same welded vertex and face arrays the graph
is built from, so the picture shows exactly the surface the solver walked.
'''

import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from geodesic import extreme_vertices, shortest_surface_path
from mesh_graph import MeshAdjacencyGraph
from obj_parser import load_obj

def welded_faces(faces, index_map):
    '''
    The file's triangles reindexed into the graph's welded vertex space.
    Triangles whose corners welded together have no area left, so they are
    dropped rather than handed to the renderer.
    '''
    triangles = []
    for face in faces:
        a, b, c = (index_map[corner] for corner in face)
        if a != b and b != c and a != c:
            triangles.append((a, b, c))
    return triangles

def cumulative_distance(graph, path):
    '''Surface distance from the origin to each vertex along the path.'''
    running = [0.0]
    for index in range(1, len(path)):
        running.append(running[-1]
                       + graph.adjacency[path[index - 1]][path[index]])
    return running

def render(mesh_path, origin=None, destination=None, out='path.png'):
    vertices, faces = load_obj(mesh_path)
    graph = MeshAdjacencyGraph(vertices, faces)
    component = graph.largest_component()

    if origin is None or destination is None:
        origin, destination = extreme_vertices(graph, component)

    distance, path = shortest_surface_path(graph, origin, destination)
    if not path:
        print(f'no surface path from {origin} to {destination}')
        return

    points = graph.vertices
    triangles = welded_faces(faces, graph.index_map)

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]

    path_x = [xs[vertex] for vertex in path]
    path_y = [ys[vertex] for vertex in path]
    path_z = [zs[vertex] for vertex in path]
    along = cumulative_distance(graph, path)

    figure = plt.figure(figsize=(17, 5.5))
    surface = figure.add_subplot(1, 3, 1, projection='3d')
    plan = figure.add_subplot(1, 3, 2)
    profile = figure.add_subplot(1, 3, 3)

    # The mesh is Y up but matplotlib draws the third argument vertically,
    # so pass X, Z, Y and the island stands upright. The path is lifted
    # clear of the terrain, though parts of it stay hidden behind higher
    # ground; the plan view is where the route is actually read.
    lift = 0.06 * (max(ys) - min(ys))
    lifted = [y + lift for y in path_y]

    # Part transparent so the stretch of path on the far side shows
    # through the cone instead of vanishing behind it.
    surface.plot_trisurf(xs, zs, ys, triangles=triangles, color='#cfd8dc',
                         linewidth=0, antialiased=True, shade=True, alpha=0.7)
    surface.plot(path_x, path_z, lifted, color='white', linewidth=5,
                 solid_capstyle='round')
    surface.plot(path_x, path_z, lifted, color='crimson', linewidth=2.5,
                 solid_capstyle='round')
    surface.view_init(elev=32, azim=-60)
    surface.set_box_aspect((1, 1, 0.55))
    surface.set_axis_off()
    surface.set_title('the mesh')

    # Plan view. The edges underneath are the graph itself, every vertex and
    # edge Dijkstra had available, with the chosen route on top.
    plan.triplot(Triangulation(xs, zs, triangles), color='#cfd8dc',
                 linewidth=0.25)
    plan.plot(path_x, path_z, color='white', linewidth=6,
              solid_capstyle='round')
    plan.plot(path_x, path_z, color='crimson', linewidth=3,
              solid_capstyle='round')
    plan.plot([path_x[0], path_x[-1]], [path_z[0], path_z[-1]], 'o',
              color='black', markersize=8, markeredgecolor='white',
              markeredgewidth=1.5)
    plan.annotate('start', (path_x[0], path_z[0]),
                  textcoords='offset points', xytext=(10, 6))
    plan.annotate('end', (path_x[-1], path_z[-1]),
                  textcoords='offset points', xytext=(10, 6))
    plan.set_aspect('equal')
    plan.set_axis_off()
    plan.set_title('the route, over the mesh graph')

    # Elevation along the route, against the summit for scale. This is part
    # of where the extra distance comes from: the path is longer than the
    # straight line because it climbs as well as detours.
    summit = max(ys)
    profile.fill_between(along, min(ys), path_y, color='#eceff1')
    profile.plot(along, path_y, color='crimson', linewidth=2.5)
    profile.plot([along[0], along[-1]], [path_y[0], path_y[-1]], 'o',
                 color='black', markersize=7, markeredgecolor='white',
                 markeredgewidth=1.2)
    profile.axhline(summit, color='#90a4ae', linestyle='--', linewidth=1)
    profile.annotate(f'summit {summit:.3f}', (along[-1], summit),
                     ha='right', va='bottom', color='#607d8b')
    profile.set_xlabel('surface distance travelled')
    profile.set_ylabel('height')
    profile.set_ylim(min(ys), summit * 1.12)
    profile.set_title('elevation along the route')
    profile.grid(alpha=0.25)

    figure.suptitle(f'{mesh_path.split("/")[-1]}: surface distance '
                    f'{distance:.4f} from vertex {origin} to {destination}, '
                    f'{len(path)} vertices')
    figure.tight_layout()
    figure.savefig(out, dpi=150)
    print(f'wrote {out}: {distance:.4f} over {len(path)} vertices, '
          f'height {min(path_y):.4f} to {max(path_y):.4f}')

# Main Execution

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    if not 1 <= len(args) <= 3:
        print('Usage: render_path.py mesh.obj [origin destination]')
        sys.exit(1)

    origin = int(args[1]) if len(args) == 3 else None
    destination = int(args[2]) if len(args) == 3 else None
    out = args[0].split('/')[-1].replace('.obj', '_path.png')
    render(args[0], origin, destination, out)

if __name__ == '__main__':
    main()
