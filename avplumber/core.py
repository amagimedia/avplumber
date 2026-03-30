import _avplumber

from _avplumber import AVPlumber as AVPlumber_C

class AVPlumber(AVPlumber_C):
    def __init__(self):
        super().__init__()

    def getEdge(self, name: str):
        return self.manager.edges.find__VideoFrame(name) or self.manager.edges.find__AudioSamples(name) or self.manager.edges.find__Packet(name) or self.manager.edges.find__EglImageFrame(name)

    @property
    def edges(self):
        return self.manager.edges

    @property
    def nodes(self):
        return self.manager.nodes

    @property
    def groups(self):
        return self.manager.groups

    @property
    def allNodes(self):
        return self.manager.allNodes
