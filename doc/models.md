# Neural model integration

AVPlumber loads prebuilt TensorRT engines; it does not own application model
weights or training datasets. Application repositories should document model
provenance, class mappings, prompts, and task-specific thresholds alongside
their graphs.

## TensorRT conversion

Export a fixed-shape ONNX model using the training framework that owns it, then
build an engine with a TensorRT version compatible with the AVPlumber runtime:

```sh
trtexec \
  --onnx=<model.onnx> \
  --saveEngine=<model.plan> \
  --fp16
```

Use explicit shape flags when the ONNX input is dynamic. Engine files are tied
to the TensorRT/CUDA environment and should normally be built for the target
deployment rather than committed.

## Graph ownership

Generic inference nodes such as `cuda_infer_yolo` and `cuda_infer_rtdetr`
accept engine paths and metadata keys as node parameters.
Model-specific graphs and class names do not belong in this framework
repository.

See `doc/NODES.md` for each node's runtime contract.
