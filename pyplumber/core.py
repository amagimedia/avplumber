import _avplumber

from _avplumber import AVPlumber as AVPlumber_C
from .node import PythonNode

_EDGE_TYPES = {
    "VideoFrame": ("exists__VideoFrame", "find__VideoFrame"),
    "AudioSamples": ("exists__AudioSamples", "find__AudioSamples"),
    "Packet": ("exists__Packet", "find__Packet"),
    "EglImageFrame": ("exists__EglImageFrame", "find__EglImageFrame"),
    "MetadataFrame": ("exists__MetadataFrame", "find__MetadataFrame"),
}

_EDGE_TYPE_ALIASES = {
    "video": "VideoFrame",
    "video_frame": "VideoFrame",
    "av::VideoFrame": "VideoFrame",
    "VideoFrame": "VideoFrame",
    "audio": "AudioSamples",
    "audio_samples": "AudioSamples",
    "av::AudioSamples": "AudioSamples",
    "AudioSamples": "AudioSamples",
    "packet": "Packet",
    "av::Packet": "Packet",
    "Packet": "Packet",
    "egl": "EglImageFrame",
    "egl_image": "EglImageFrame",
    "EglImageFrame": "EglImageFrame",
    "metadata": "MetadataFrame",
    "metadata_frame": "MetadataFrame",
    "MetadataFrame": "MetadataFrame",
}


def _edge_type_name(data_type):
    if data_type is None:
        return "VideoFrame"
    key = _EDGE_TYPE_ALIASES.get(str(data_type))
    if key is None:
        raise ValueError(f"Unsupported edge data type: {data_type}")
    return key


class AVPlumber(AVPlumber_C):
    def __init__(self):
        super().__init__()

    def getEdge(self, name: str, data_type=None):
        for exists_name, find_name in _EDGE_TYPES.values():
            if getattr(self.manager.edges, exists_name)(name):
                return getattr(self.manager.edges, find_name)(name)

        _, find_name = _EDGE_TYPES[_edge_type_name(data_type)]
        return getattr(self.manager.edges, find_name)(name)

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
