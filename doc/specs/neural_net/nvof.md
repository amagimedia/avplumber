# neural_net/nvof — NVIDIA Optical Flow Frame Interpolation

## Node: `nvof_fruc`
Frame Rate Up Conversion via NVIDIA Optical Flow SDK.

### Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `fruc_library_path` | required | Path to libNvOFFRUC.so |
| `passthrough_on_fail` | true | Pass through frames if FRUC fails |
| `factor` | 2 | Frame-rate multiplication factor (2–16). `factor=N` emits N-1 interpolated frames per input pair, i.e. `factor=4` → 4× fps (3 synthesized + 1 original per input). For super-slomo pair with `speed_video { "speed": 1/N }` to preserve realtime playback. |

### Pipeline
1. Load NvOFFRUC library via dlopen
2. Allocate CUDA arrays for NV12 frame pairs
3. Register arrays with FRUC engine
4. For each input frame pair: compute optical flow, interpolate intermediate frame
5. Output: 2× frame rate (interpolated frame + original)

### Requirements
- Build: `HAVE_CUDA=1 HAVE_NVOF_FRUC=1` + Optical_Flow_SDK_5.0.7 headers
- Runtime: libNvOFFRUC.so (loaded dynamically)
