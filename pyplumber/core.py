import _avplumber

from _avplumber import AVPlumber as AVPlumber_C
from .node import PythonNode

class AVPlumber(AVPlumber_C):
    def __init__(self):
        super().__init__()

    def getEdge(self, name: str):
        return self.manager.edges.find__VideoFrame(name) or self.manager.edges.find__AudioSamples(name) or self.manager.edges.find__Packet(name) or self.manager.edges.find__EglImageFrame(name)

    def addNode(self, node:"NodeBase", early_create: bool = False, start: bool = False):
        added_node = self.manager.addNode(node.parameters, early_create, start, node if isinstance(node, PythonNode) else None)
        node._wrapper = added_node
        node.add_to_avplumber(self)

    @property
    def edges(self):
        return self.manager.edges

    def node(self, name: str):
        return self.manager.node(name)

    def nodes(self, type_name: str):
        return self.manager.nodes(type_name)

    def group(self, name: str):
        return self.manager.group(name)

    @property
    def allNodes(self):
        return self.manager.allNodes
