import tensorrt as trt, os, time

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)

with open("/root/rife426/rife_v4.26_fp16.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print("PARSE ERR:", parser.get_error(i))
        raise SystemExit(1)

config = builder.create_builder_config()

config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 * 1024 * 1024 * 1024)

profile = builder.create_optimization_profile()
# Fixed at 1088x1920 (1080p padded to next mult of 32). H must be mult of 32.
profile.set_shape("img0", (1, 3, 1088, 1920), (1, 3, 1088, 1920), (1, 3, 1088, 1920))
profile.set_shape("img1", (1, 3, 1088, 1920), (1, 3, 1088, 1920), (1, 3, 1088, 1920))
profile.set_shape("timestep", (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))
config.add_optimization_profile(profile)

print("Building engine (fp16, 1088x1920)...")
t0 = time.time()
plan = builder.build_serialized_network(network, config)
if plan is None:
    raise SystemExit("build failed")
print(f"Built in {time.time()-t0:.1f}s, size={plan.nbytes/1e6:.1f} MB")

out = "/root/rife426/rife_v4.26_fp16_1080p.engine"
with open(out, "wb") as f:
    f.write(bytes(plan))
print("wrote", out)
