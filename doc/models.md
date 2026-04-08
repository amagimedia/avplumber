# YOLO / TensorRT Models Inventory

Remote host: `fedora@172.17.36.132:/home/fedora/tensorrt/`

## Environment

| Component | Version / Path |
|-----------|---------------|
| GPU | Tesla T4 (15 GB) |
| CUDA | 13.0 (`/usr/local/cuda-13.0`) |
| TensorRT | 10.15.1 (`/opt/tensorrt/`) |
| trtexec | `/opt/tensorrt/bin/trtexec` |
| Ultralytics | 8.4.19 |
| Python venv | `/home/fedora/tensorrt/.venv` (Python 3.13) |

---

## Models

### ball (1 class: basketball)

Single-class ball detection. Dataset: `nba-ball/data.yaml`.

| Stage | Path | Size | Notes |
|-------|------|------|-------|
| .pt | `nba-ball/yolo26s.pt` | 20.4 MB | pretrained base |
| .pt | `ball.pt` | 20.3 MB | best.pt copy (yolo26s) |
| .onnx | `nba-ball/yolo26s.onnx` | 38.3 MB | default res |
| .onnx | `ball_960x544.onnx` | 38.2 MB | 960x544 |
| .plan | `nba-ball/yolo26s.plan` | 21.4 MB | default res |
| .plan | `ball.plan` | 21.3 MB | default res |
| .plan | `ball_960x544.plan` | 21.3 MB | 960x544 |

Training run: `nba-ball/runs/detect/runs/train/nba_ball_yolo26s3/`

### hoop-ball (2 classes: ball, hoop)

Dataset: `basketballv2iyolo26` (Ultralytics Hub, mthunzi-musumadi). Base: **yolo26n**.

| Stage | Path | Size |
|-------|------|------|
| .pt | `hoop-ball.pt` | 5.5 MB |
| .onnx | `hoop-ball.onnx` | 10.8 MB |
| .plan | `hoop-ball.plan` | 7.4 MB |

### player-ball (2 classes: ball, player)

Large model. Base: **yolo26x**. Training run: `tomer-yadgarov/exp-16`.

| Stage | Path | Size |
|-------|------|------|
| .pt | `player-ball.pt` | 53.0 MB |
| .onnx | `player-ball.onnx` | 99.6 MB |
| .plan | `player-ball.plan` | 52.1 MB |

### player-ball-2cls (2 classes: ball, player)

Base: **yolo11s**. Trained 200 epochs, imgsz 960.

| Stage | Path | Size | Resolution |
|-------|------|------|------------|
| .pt | `player-ball-2cls/player-ball-2cls.pt` | 19.2 MB | - |
| .onnx | `player-ball-2cls/player-ball-2cls_960x544.onnx` | 38.0 MB | 960x544 |
| .onnx | `player-ball-2cls/player-ball-2cls_960x960.onnx` | 38.2 MB | 960x960 |
| .plan | `player-ball-2cls/player-ball-2cls.plan` | 20.9 MB | default |
| .plan | `player-ball-2cls/player-ball-2cls_960x544.plan` | 20.9 MB | 960x544 |

### player-ball-3cls (3 classes: ball, foot, player)

Base: **yolo11s**. Trained 200 epochs, imgsz 960. Dataset: `basketball-player-detection-2v7iyolo26`.

| Stage | Path | Size | Resolution |
|-------|------|------|------------|
| .pt | `player-ball-3cls/player-ball-3cls.pt` | 19.2 MB | - |
| .onnx | `player-ball-3cls/player-ball-3cls_960x960.onnx` | 38.2 MB | 960x960 |
| .onnx | `player-ball-3cls/player-ball-3cls_960x544.onnx` | 38.0 MB | 960x544 |
| .plan | `player-ball-3cls/player-ball-3cls.plan` | 20.9 MB | default |
| .plan | `player-ball-3cls/player-ball-3cls_960x544.plan` | 20.9 MB | 960x544 |

### basketball-players-full (9 classes)

Classes: Ball, Hoop, Period, Player, Ref, Shot Clock, Team Name, Team Points, Time Remaining.
Dataset: Roboflow `basketball-players-fy4c2` v25 (1140 train / 32 val images).
Base: **yolo26s**. Training: 100 epochs, imgsz 960, batch 16. **Trained 2025-04-07**.

| Stage | Path | Size | Notes |
|-------|------|------|-------|
| .pt | `basketball-players-full/train2/weights/best.pt` | 20 MB | mAP50=0.901, mAP50-95=0.669 |
| .onnx | `basketball-players-full/train2/weights/best.onnx` | 36.5 MB | 960x544 |
| .plan | `basketball-players-full/basketball-players-full_960x544.plan` | 20 MB | FP16, 305 fps / 3.3ms on T4 |

### court-segmentation (2 classes: basketball-court, three-point-line)

Segmentation model. Base: **yolo26s-seg**. Dataset: `court-segmentationv1iyolo26` (Ultralytics Hub, tomer-yadgarov).

| Stage | Path | Size |
|-------|------|------|
| .pt | `court-segmentation.pt` | 23.3 MB |
| .onnx | `court-segmentation.onnx` | 42.0 MB |
| .plan | `court-segmentation.plan` | 23.3 MB |

### court-pose (1 class: court, 33 keypoints)

Pose/keypoint model for court detection. Dataset: Roboflow `basketball-court-detection-2` v19. kpt_shape: [33, 3].

| Stage | Path | Size |
|-------|------|------|
| .pt | `court-pose-4/train4/weights/best.pt` | 163.8 MB |
| .onnx | `court-pose-4/court-pose.onnx` | 109.5 MB |
| .plan | `court-pose-4/court-pose.plan` | 110.8 MB |

### amagi-reframer

Non-YOLO reframer models.

| Stage | Path | Size |
|-------|------|------|
| .onnx | `amagi-reframer.onnx` | 21.2 MB |
| .plan | `amagi-reframer.plan` | 11.2 MB |
| .onnx | `amagi-reframer2.onnx` | 33.1 MB |
| .plan | `amagi-reframer2.plan` | 17.3 MB |

---

## Base Weights (pretrained, not fine-tuned)

| Path | Size | Architecture |
|------|------|-------------|
| `yolo26n.pt` | 5.5 MB | YOLOv26 nano |
| `nba-ball/yolo26s.pt` | 20.4 MB | YOLOv26 small |
| `weights/ultralytics/yolo26/yolo26l/yolo26l.pt` | 53.2 MB | YOLOv26 large |
| `yolo26x.pt` | 118.7 MB | YOLOv26 extra-large |
| `yolo26s-seg.pt` | 23.5 MB | YOLOv26 small (segmentation) |
| `yolo11s.pt` | 19.3 MB | YOLO11 small |
| `weights/ultralytics/yolo11/yolo11n/yolo11n.pt` | 5.6 MB | YOLO11 nano |
| `yolo11x.pt` | 114.6 MB | YOLO11 extra-large |

---

## TensorRT CLI Commands

All commands run on the remote host. Activate venv first:

```bash
source /home/fedora/tensorrt/.venv/bin/activate
```

### 1. Train (.pt)

```bash
yolo detect train \
  model=yolo26s.pt \
  data=/path/to/data.yaml \
  epochs=100 \
  imgsz=960 \
  batch=16 \
  project=/home/fedora/tensorrt/<model-name> \
  name=train
```

For segmentation:
```bash
yolo segment train model=yolo26s-seg.pt data=/path/to/data.yaml ...
```

For pose:
```bash
yolo pose train model=yolo26x-pose.pt data=/path/to/data.yaml ...
```

### 2. Export to ONNX

Default resolution (square):
```bash
yolo export model=best.pt format=onnx opset=13 simplify=True
```

Specific resolution (e.g. 960x544 for 16:9 video):
```bash
yolo export model=best.pt format=onnx imgsz=544,960 opset=13 simplify=True
```

Note: `imgsz` order is `height,width`.

### 3. Convert ONNX to TensorRT .plan

```bash
/opt/tensorrt/bin/trtexec \
  --onnx=/path/to/model.onnx \
  --saveEngine=/path/to/model.plan \
  --fp16
```

With explicit input shape (dynamic batch):
```bash
/opt/tensorrt/bin/trtexec \
  --onnx=/path/to/model.onnx \
  --saveEngine=/path/to/model.plan \
  --fp16 \
  --minShapes=images:1x3x544x960 \
  --optShapes=images:1x3x544x960 \
  --maxShapes=images:1x3x544x960
```

### Full pipeline example (basketball-players-full)

```bash
# 1. Train
yolo detect train \
  model=/home/fedora/tensorrt/nba-ball/yolo26s.pt \
  data=/home/fedora/tensorrt/basketball-players-full/data.yaml \
  epochs=100 imgsz=960 batch=16 \
  project=/home/fedora/tensorrt/basketball-players-full name=train

# 2. Export to ONNX (960x544 for 16:9)
yolo export \
  model=/home/fedora/tensorrt/basketball-players-full/train2/weights/best.pt \
  format=onnx imgsz=544,960 opset=13 simplify=True

# 3. Convert to TensorRT
/opt/tensorrt/bin/trtexec \
  --onnx=/home/fedora/tensorrt/basketball-players-full/train2/weights/best.onnx \
  --saveEngine=/home/fedora/tensorrt/basketball-players-full/basketball-players-full_960x544.plan \
  --fp16
```
