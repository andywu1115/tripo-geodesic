import doubly_linked_list as dll

class PriorityQueue:
    def __init__(self, data=None):
        self.dll = dll.DoublyLinkedList()
        if data is not None:
            for value in data:
                self.push(value)

    def push(self, data):
        self.dll.insert_ordered(data)

    def pop(self):
        return self.dll.pop(-1)

    def peek(self):
        return self.dll[-1]

    def __str__(self):
        return str(self.dll)

    def __bool__(self):
        return bool(self.dll)

    def __len__(self):
        return len(self.dll)
