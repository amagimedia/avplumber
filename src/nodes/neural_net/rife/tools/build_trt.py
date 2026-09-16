import tensorrt as trt, os, time

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)

# Must match the resolution the ONNX was exported at (see export_onnx_fp16.py).
H = int(os.environ.get("RIFE_H", 1088))
W = int(os.environ.get("RIFE_W", 1920))
onnx_path = os.environ.get("RIFE_ONNX", "/root/rife426/rife_v4.26_fp16.onnx")
out = os.environ.get("RIFE_ENGINE", "/root/rife426/rife_v4.26_fp16_1080p.engine")

with open(onnx_path, "rb") as f:
    # Pass `path` so TensorRT resolves the external-weights .data file next to
    # the .onnx instead of relative to the current working directory.
    if not parser.parse(f.read(), path=onnx_path):
        for i in range(parser.num_errors):
            print("PARSE ERR:", parser.get_error(i))
        raise SystemExit(1)

config = builder.create_builder_config()

config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 * 1024 * 1024 * 1024)

profile = builder.create_optimization_profile()
# One fixed shape (min == opt == max); H and W must be multiples of 64.
shape = (1, 3, H, W)
profile.set_shape("img0", shape, shape, shape)
profile.set_shape("img1", shape, shape, shape)
profile.set_shape("timestep", (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))
config.add_optimization_profile(profile)

print(f"Building engine (fp16, {H}x{W}) from {onnx_path}...")
t0 = time.time()
plan = builder.build_serialized_network(network, config)
if plan is None:
    raise SystemExit("build failed")
print(f"Built in {time.time()-t0:.1f}s, size={plan.nbytes/1e6:.1f} MB")

with open(out, "wb") as f:
    f.write(bytes(plan))
print("wrote", out)
