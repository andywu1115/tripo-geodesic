#!/usr/bin/env python3

'''
min_heap.py

Array-backed binary min-heap, extended from my HW09 heap.py to serve as the
frontier for Dijkstra's algorithm on a 3D mesh surface graph.

HW09 stored plain integer character frequencies for Huffman coding. Here each
item is an (distance, vertex) tuple with a float distance. Python compares
tuples element by element, so percolate_up and percolate_down are unchanged
from HW09: the smallest distance reaches the root, and equal distances are
broken by the vertex number.
'''

# Types

# One item in the heap: the total distance from the origin vertex, and the
# vertex that distance belongs to.
Entry = tuple[float, int]

class MinHeap:
    # Constructor for an empty heap
    def __init__(self) -> None:
        self.heap_array: list[Entry] = []

    # Move the item at node_index up toward the root while it is smaller
    # than its parent, restoring the min heap property after an insert
    def percolate_up(self, node_index: int) -> None:
        while node_index > 0:
            parent_index = (node_index - 1) // 2
            if self.heap_array[node_index] >= self.heap_array[parent_index]:
                return
            else:
                temp = self.heap_array[node_index]
                self.heap_array[node_index] = self.heap_array[parent_index]
                self.heap_array[parent_index] = temp
                node_index = parent_index

    # Move the item at node_index down toward the leaves while either child
    # is smaller, restoring the min heap property after a remove
    def percolate_down(self, node_index: int) -> None:
        child_index = 2 * node_index + 1
        value = self.heap_array[node_index]

        while child_index < len(self.heap_array):
            min_value = value
            min_index = -1
            i = 0
            while i < 2 and i + child_index < len(self.heap_array):
                if self.heap_array[i + child_index] < min_value:
                    min_value = self.heap_array[i + child_index]
                    min_index = i + child_index
                i = i + 1

            if min_value == value:
                return
            else:
                temp = self.heap_array[node_index]
                self.heap_array[node_index] = self.heap_array[min_index]
                self.heap_array[min_index] = temp
                node_index = min_index
                child_index = 2 * node_index + 1

    # Add an item to the heap in O(log N)
    def insert(self, value: Entry) -> None:
        self.heap_array.append(value)
        self.percolate_up(len(self.heap_array) - 1)

    # Remove and return the item with the smallest distance, in O(log N)
    # Raises IndexError on an empty heap; Dijkstra never hits this because
    # its loop tests len() first
    def remove(self) -> Entry:
        min_value = self.heap_array[0]
        replace_value = self.heap_array.pop()
        if len(self.heap_array) > 0:
            self.heap_array[0] = replace_value
            self.percolate_down(0)
        return min_value

    # Number of items in the heap, supporting len(heap)
    def __len__(self) -> int:
        return len(self.heap_array)
