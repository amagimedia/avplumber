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

Shape is fixed at 1088×1920 (1080p padded to a multiple of 32). Non-1080p
inputs need a re-export with the appropriate H/W and a new engine.

## 3. Build the TensorRT engine

```
pip install tensorrt
python3 build_trt.py
```

Output: `rife_v4.26_fp16_1080p.engine` (~420 MB). Point `rife_vfi`'s `engine`
parameter at this file. Build takes ~35 s on Ada; cache and reuse.
