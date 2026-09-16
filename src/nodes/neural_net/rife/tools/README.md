# RIFE 4.26 engine build tools

Two-step pipeline: PyTorch checkpoint → ONNX → TensorRT engine.

## 1. Fetch the checkpoint

```
curl -L -o rife426.zip https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip
unzip rife426.zip
```

Layout expected by `export_onnx_fp16.py`:

```
/root/rife426/RIFEv4.26_0921/{IFNet_HDv3.py, flownet.pkl, refine.py, RIFE_HDv3.py}
```

## 2. Export ONNX (fp16)

```
pip install torch onnx onnxscript Pillow
mkdir -p /root/rife426/train_log
ln -sf ../RIFEv4.26_0921/IFNet_HDv3.py /root/rife426/train_log/IFNet_HDv3.py
python3 export_onnx_fp16.py
```

Output: `rife_v4.26_fp16.onnx` + `rife_v4.26_fp16.onnx.data` (24 MB).

The export script inlines a corrected `warp` shim. The original Practical-RIFE
`warp` isn't importable as a module, so the shim replaces it. The important
detail: the sampling grid is precomputed in normalized `[-1, 1]` space and the
flow is normalized to the same scale before being added — mixing pixel-space
grid with normalized flow silently zeros the flow contribution and degrades
the model to a plain alpha blend between img0 and img1.

Shape defaults to 1088×1920 (1080p padded). Dimensions must be multiples of
**64**, not 32: IFNet's coarsest pyramid level is 1/16 and its blocks
downsample by a further 4, so a multiple-of-32-only size like 1440 fails at
export with `size of tensor a (1440) must match tensor b (1472)`. 1080p is
unaffected because 1088 is already 64×17.

Non-1080p inputs need a re-export with the appropriate H/W and a new engine —
both scripts read `RIFE_H`, `RIFE_W`, `RIFE_ONNX` and (for `build_trt.py`)
`RIFE_ENGINE` from the environment, so for a 1440p source (1440 → 1472):

```
RIFE_H=1472 RIFE_W=2560 RIFE_ONNX=/root/rife426/rife_v4.26_fp16_1440p.onnx \
  python3 export_onnx_fp16.py
RIFE_H=1472 RIFE_W=2560 RIFE_ONNX=/root/rife426/rife_v4.26_fp16_1440p.onnx \
  RIFE_ENGINE=/root/rife426/rife_v4.26_fp16_1440p.engine python3 build_trt.py
```

`rife_vfi` reads the expected resolution off the engine, so pointing its
`engine` param at the new file is all that's needed.

## 3. Build the TensorRT engine

```
pip install tensorrt
python3 build_trt.py
```

Output: `rife_v4.26_fp16_1080p.engine` (~420 MB). Point `rife_vfi`'s `engine`
parameter at this file. Build takes ~35 s on Ada; cache and reuse.
