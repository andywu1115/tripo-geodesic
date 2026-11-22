#!/usr/bin/env python3

'''
obj_parser_test.py

Unit tests for the .obj parser, using io.StringIO as a mock file so no mesh
on disk is needed.
'''

import io
import unittest
from obj_parser import read_obj, triangulate

class ObjParserTest(unittest.TestCase):

    def test_single_triangle(self):
        text = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n'
        vertices, faces = read_obj(io.StringIO(text))
        self.assertEqual(vertices, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
        self.assertEqual(faces, [(0, 1, 2)])

    def test_coordinates_are_floats(self):
        vertices, _ = read_obj(io.StringIO('v -1.5 2 3e2\n'))
        self.assertEqual(vertices, [(-1.5, 2.0, 300.0)])
        self.assertIsInstance(vertices[0][0], float)

    def test_vertex_colors_are_ignored(self):
        # Tripo writes "v x y z r g b"; only the position is read
        vertices, _ = read_obj(io.StringIO('v 1 2 3 0.5 0.25 0.125\n'))
        self.assertEqual(vertices, [(1.0, 2.0, 3.0)])

    def test_face_index_fields(self):
        # "v/vt/vn" and "v//vn" both carry the vertex before the first slash
        text = ('v 0 0 0\nv 1 0 0\nv 0 1 0\n'
                'f 1/1/1 2/2/2 3/3/3\n'
                'f 1//1 2//2 3//3\n')
        _, faces = read_obj(io.StringIO(text))
        self.assertEqual(faces, [(0, 1, 2), (0, 1, 2)])

    def test_quad_becomes_two_triangles(self):
        text = 'v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n'
        _, faces = read_obj(io.StringIO(text))
        self.assertEqual(faces, [(0, 1, 2), (0, 2, 3)])

    def test_triangulate_counts(self):
        # An n-gon becomes n - 2 triangles
        self.assertEqual(len(triangulate([0, 1, 2])), 1)
        self.assertEqual(len(triangulate([0, 1, 2, 3])), 2)
        self.assertEqual(len(triangulate([0, 1, 2, 3, 4])), 3)

    def test_negative_index_counts_backward(self):
        # -1 is the most recent vertex, so this is the same triangle as "f 1 2 3"
        text = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n'
        _, faces = read_obj(io.StringIO(text))
        self.assertEqual(faces, [(0, 1, 2)])

    def test_other_records_are_ignored(self):
        text = ('# a comment\n'
                '\n'
                'mtllib mesh.mtl\n'
                'o head\n'
                'v 0 0 0\nv 1 0 0\nv 0 1 0\n'
                'vt 0.5 0.5\nvn 0 0 1\ns off\nusemtl skin\n'
                'f 1 2 3\n')
        vertices, faces = read_obj(io.StringIO(text))
        self.assertEqual(len(vertices), 3)
        self.assertEqual(faces, [(0, 1, 2)])

    def test_empty_file(self):
        vertices, faces = read_obj(io.StringIO(''))
        self.assertEqual(vertices, [])
        self.assertEqual(faces, [])

    def test_index_zero_raises(self):
        with self.assertRaises(ValueError):
            read_obj(io.StringIO('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n'))

    def test_index_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            read_obj(io.StringIO('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n'))

    def test_short_face_raises(self):
        with self.assertRaises(ValueError):
            read_obj(io.StringIO('v 0 0 0\nv 1 0 0\nf 1 2\n'))

    def test_short_vertex_raises(self):
        with self.assertRaises(ValueError):
            read_obj(io.StringIO('v 1 2\n'))

if __name__ == '__main__':
    main_unused = None
    unittest.main()
