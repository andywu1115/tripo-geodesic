#!/usr/bin/env python3

'''
min_heap_test.py

Unit tests for the MinHeap used as the Dijkstra frontier: the heap invariant
after every operation, tuple ordering, float distances, popping until empty,
and the empty-heap error.
'''

import unittest
from min_heap import MinHeap

class MinHeapTest(unittest.TestCase):

    # Assert that every parent is less than or equal to both of its children.
    # Checking the whole array, not just the root, catches a percolate that
    # stops one level too early.
    def check_heap_property(self, heap):
        for parent_index in range(len(heap.heap_array)):
            for child_index in (2 * parent_index + 1, 2 * parent_index + 2):
                if child_index < len(heap.heap_array):
                    self.assertLessEqual(heap.heap_array[parent_index],
                                         heap.heap_array[child_index])

    def test_insert_keeps_heap_property(self):
        heap = MinHeap()
        for entry in [(3.0, 7), (1.5, 2), (4.25, 1), (0.5, 9), (2.0, 4)]:
            heap.insert(entry)
            self.check_heap_property(heap)
        self.assertEqual(len(heap), 5)
        self.assertEqual(heap.heap_array[0], (0.5, 9))

    def test_remove_keeps_heap_property(self):
        heap = MinHeap()
        for entry in [(3.0, 7), (1.5, 2), (4.25, 1), (0.5, 9), (2.0, 4)]:
            heap.insert(entry)
        while len(heap) > 0:
            heap.remove()
            self.check_heap_property(heap)

    def test_pop_until_empty_is_non_decreasing(self):
        entries = [(5.5, 1), (0.25, 2), (9.0, 3), (3.75, 4), (1.0, 5),
                   (7.125, 6), (2.5, 7), (8.0, 8), (4.0, 9), (6.0, 10)]
        heap = MinHeap()
        for entry in entries:
            heap.insert(entry)
        popped = []
        while len(heap) > 0:
            popped.append(heap.remove())
        self.assertEqual(popped, sorted(entries))

    def test_ties_broken_by_vertex_number(self):
        heap = MinHeap()
        for entry in [(2.0, 5), (2.0, 3), (2.0, 8)]:
            heap.insert(entry)
        self.assertEqual(heap.remove(), (2.0, 3))
        self.assertEqual(heap.remove(), (2.0, 5))
        self.assertEqual(heap.remove(), (2.0, 8))

    def test_float_distances_compare_exactly(self):
        # 0.1 + 0.2 is 0.30000000000000004 in binary floating point, so it is
        # strictly greater than 0.3 and must come out second
        heap = MinHeap()
        heap.insert((0.1 + 0.2, 1))
        heap.insert((0.3, 2))
        self.assertEqual(heap.remove(), (0.3, 2))
        self.assertEqual(heap.remove(), (0.1 + 0.2, 1))

    def test_len_tracks_size(self):
        heap = MinHeap()
        self.assertEqual(len(heap), 0)
        heap.insert((1.0, 1))
        heap.insert((2.0, 2))
        self.assertEqual(len(heap), 2)
        heap.remove()
        self.assertEqual(len(heap), 1)

    def test_remove_from_empty_raises(self):
        heap = MinHeap()
        with self.assertRaises(IndexError):
            heap.remove()

if __name__ == '__main__':
    unittest.main()
