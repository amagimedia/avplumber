# neural_net/tracking — Generic Tracking Nodes

## `player_tracker`

Applies ByteTrack to detections in a configurable metadata key. Target labels,
confidence thresholds, buffer length, and camera-cut reset metadata are graph
parameters; the node does not assume a sport.

## `tracknet_ball`

Runs a temporal TrackNet TensorRT engine over CUDA NV12 frames. It can emit a
compact ball detection, raw engine tensors, or both. Model paths, labels,
normalization, temporal alignment, and metadata keys are supplied by the
downstream application.

See `doc/NODES.md` for the complete TrackNet parameter contract.
