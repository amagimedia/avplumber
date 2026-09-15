# neural_net/nvof — NVIDIA Optical Flow Frame Interpolation

## Node: `nvof_fruc`
Frame Rate Up Conversion via NVIDIA Optical Flow SDK.

### Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `fruc_library_path` | required | Path to libNvOFFRUC.so |
| `passthrough_on_fail` | true | Pass through frames if FRUC fails |
| `factor` | 2 | Frame-rate multiplication factor (2–9). `factor=N` emits N-1 interpolated frames per input pair, i.e. `factor=4` → 4× fps (3 synthesized + 1 original per input). For super-slomo pair with `speed_video { "speed": 1/N }` to preserve realtime playback. **For factor > 2 prefer cascading multiple factor=2 nodes** (see `examples/super_slomo_cascade.avplumber`): FRUC's flow refinement is tuned for the midpoint output, so a single factor=4 stage requesting t={0.25, 0.5, 0.75} gets lower-quality flow on the off-midpoint samples than two factor=2 stages in series. Same total FRUC work, better quality, and the two stages pipeline so throughput is unchanged or slightly better. Max is 9 (registered CUarrays: 2 render + up to 8 interp = NvOFFRUC_MAX_RESOURCE). |

### Pipeline
1. Load NvOFFRUC library via dlopen
2. Allocate CUDA arrays for NV12 frame pairs
3. Register arrays with FRUC engine
4. For each input frame pair: compute optical flow, interpolate intermediate frame
5. Output: 2× frame rate (interpolated frame + original)

### Requirements
- Build: `HAVE_CUDA=1 HAVE_NVOF_FRUC=1` + Optical_Flow_SDK_5.0.7 headers
- Runtime: libNvOFFRUC.so (loaded dynamically)
