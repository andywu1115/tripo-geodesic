#!/usr/bin/env python3

'''
obj_parser.py

Read a Wavefront .obj mesh into a vertex list and a triangle list without a
mesh library, so the surface graph is built from the file's own geometry.

The format is plain text, one record per line. The two records that matter
here are "v x y z" for a vertex position and "f ..." for a face. Every other
keyword (vt, vn, g, o, s, usemtl, mtllib) describes texture, shading or
grouping, none of which affects distances on the surface, so it is ignored.
'''

import sys

# Types

# A vertex position in 3D, from a "v x y z" line
Vertex = tuple[float, float, float]

# A triangle, as three indices into the vertex list, numbered from 0
Face = tuple[int, int, int]

# Functions

def parse_vertex(fields: list[str], line_number: int) -> Vertex:
    '''
    Turn the fields after "v" into an (x, y, z) tuple of floats.
    Only the first three are read: an .obj vertex may carry a fourth
    coordinate, or three more numbers for a vertex color, and Tripo's
    exports often include the color.
    '''
    if len(fields) < 3:
        raise ValueError(f'line {line_number}: a vertex needs 3 coordinates')
    return (float(fields[0]), float(fields[1]), float(fields[2]))

def parse_face_index(field: str, vertex_count: int, line_number: int) -> int:
    '''
    Turn one field of an "f" line into an index into the vertex list.
    A field is "v", "v/vt", "v//vn" or "v/vt/vn", so the vertex is the part
    before the first slash. The file numbers vertices from 1, and a negative
    number counts backward from the most recent vertex, so both are
    converted to an index from 0 here.
    '''
    index = int(field.split('/')[0])
    if index > 0:
        index = index - 1
    elif index < 0:
        index = vertex_count + index
    else:
        raise ValueError(f'line {line_number}: vertex index 0 is not valid')

    if index < 0 or index >= vertex_count:
        raise ValueError(f'line {line_number}: vertex index out of range')
    return index

def triangulate(indices: list[int]) -> list[Face]:
    '''
    Split one face into triangles with a fan from its first vertex.
    A triangle comes back unchanged, a quad becomes two triangles, and an
    n-gon becomes n - 2. Every triangle keeps the first vertex, so the fan
    covers the face with no gaps and no overlaps for a convex face.
    '''
    return [(indices[0], indices[i], indices[i + 1])
            for i in range(1, len(indices) - 1)]

def read_obj(stream) -> tuple[list[Vertex], list[Face]]:
    '''
    Read a mesh from a stream of .obj text.
    Returns the vertex list and the triangle list. Vertices keep the order
    they appear in the file, because face records index them by position.
    '''
    vertices: list[Vertex] = []
    faces: list[Face] = []

    for line_number, line in enumerate(stream, start=1):
        fields = line.split()

        # Skip blank lines and comments
        if not fields or fields[0].startswith('#'):
            continue

        if fields[0] == 'v':
            vertices.append(parse_vertex(fields[1:], line_number))
        elif fields[0] == 'f':
            if len(fields) < 4:
                raise ValueError(
                    f'line {line_number}: a face needs at least 3 vertices')
            indices = [parse_face_index(field, len(vertices), line_number)
                       for field in fields[1:]]
            faces.extend(triangulate(indices))

    return vertices, faces

def load_obj(path: str) -> tuple[list[Vertex], list[Face]]:
    '''Read a mesh from a file path.'''
    with open(path) as stream:
        return read_obj(stream)

# Main Execution

def main(args=None) -> None:
    if args is None:
        args = sys.argv[1:]
    if len(args) != 1:
        print('Usage: obj_parser.py mesh.obj')
        sys.exit(1)

    vertices, faces = load_obj(args[0])
    print(f'{len(vertices):,} vertices, {len(faces):,} triangles')

    if vertices:
        for axis, name in enumerate('xyz'):
            values = [vertex[axis] for vertex in vertices]
            print(f'{name}: {min(values):.4f} to {max(values):.4f}')

if __name__ == '__main__':
    main()
