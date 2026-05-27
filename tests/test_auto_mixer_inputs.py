from pyplumber.auto_mixer.inputs import build_input_subgraph


class FakeAVPlumber:
    def __init__(self):
        self.nodes = []

    def addNode(self, node):
        self.nodes.append(node)


def _yolo_node(avp):
    matches = [n for n in avp.nodes if n.parameters.get("type") == "cuda_infer_yolo"]
    assert len(matches) == 1
    return matches[0]


def _node_by_name(avp, name):
    matches = [n for n in avp.nodes if n.parameters.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def test_input_subgraph_leaves_cuda_graph_disabled_by_default():
    avp = FakeAVPlumber()

    build_input_subgraph(avp, 0, "input.ts", "face.plan")

    assert _yolo_node(avp).parameters["use_cuda_graph"] is False


def test_input_subgraph_can_enable_cuda_graph_for_face_inference():
    avp = FakeAVPlumber()

    build_input_subgraph(avp, 0, "input.ts", "face.plan", face_use_cuda_graph=True)

    assert _yolo_node(avp).parameters["use_cuda_graph"] is True


def test_input_subgraph_always_tracks_yolo_face_for_portrait_crop():
    avp = FakeAVPlumber()

    build_input_subgraph(avp, 0, "input.ts", "face.plan")

    assert _node_by_name(avp, "tracker_0").parameters["target_labels"] == ["Face"]
    assert _node_by_name(avp, "face_crop_0").parameters["metadata_key"] == (
        "smoothed_crop_viewport_v1"
    )
    assert not any(n.parameters.get("name") == "static_vp_0" for n in avp.nodes)
