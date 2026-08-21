from .mbr import MBR
from .node import Node


class Entry:

    def __init__(self, mbr: MBR = None, child: Node | int = None, node: Node = None):
        self.mbr = mbr
        self.child = child
        self.node = node

    @property
    def priority(self):
        return self.mbr.max_dim

    def __lt__(self, other):
        return tuple(self.priority) > tuple(other.priority)

    def __gt__(self, other):
        return tuple(self.priority) < tuple(other.priority)
