import heapq

import numpy as np

from .node import Node


class Heap:

    def __init__(self):
        # initialize the heap
        self.heap = []

    def push(self, node: Node):
        for entry in node.entries:
            heapq.heappush(self.heap, entry)

    def pop(self):
        return heapq.heappop(self.heap)

    def peek(self):
        return self.heap[0] if self.heap else None

    def empty(self) -> bool:
        return self.__len__() == 0

    def __len__(self):
        return len(self.heap)
