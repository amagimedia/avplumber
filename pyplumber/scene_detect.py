"""CUDA-accelerated scene detection nodes for AVPlumber.

Provides scene change detection running directly on CUDA frames without CPU round-trips.
Interface inspired by PySceneDetect but optimized for GPU processing.
"""

import json
import sys
import time
from typing import Optional, List, Tuple

try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    cp = None
    HAVE_CUPY = False

from pyplumber.node import PythonNode


class SceneDetector:
    """Base class for scene detectors. Subclasses implement specific algorithms."""

    def __init__(self, threshold: float, min_scene_len: int = 15, hist_bins: int = 256,
                 confirm_frames: int = 3):
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.hist_bins = hist_bins
        self.confirm_frames = confirm_frames
        self._frames_since_cut = 0
        self._last_state = None
        self._streak = 0  # consecutive over-threshold frames

    def reset(self):
        """Reset detector state."""
        self._frames_since_cut = 0
        self._last_state = None
        self._streak = 0

    # Return values from _check_cut:
    #   "no_cut"   — normal frame, forward immediately
    #   "pending"  — streak started/continuing, buffer this frame
    #   "confirm"  — streak complete, the cut is at the FIRST buffered frame
    #   "abort"    — streak broke, flush buffered frames as normal (no cut)
    CUT_NO     = "no_cut"
    CUT_PEND   = "pending"
    CUT_CONF   = "confirm"
    CUT_ABORT  = "abort"

    def _check_cut(self, over_threshold: bool) -> str:
        """Apply confirm_frames debounce. Returns one of the CUT_* constants.

        State frozen during streak so subsequent frames compare against the
        pre-cut reference until confirmation or abort.
        """
        self._frames_since_cut += 1
        if over_threshold and self._frames_since_cut > self.min_scene_len:
            self._streak += 1
            self._freeze_state = True
            if self._streak >= self.confirm_frames:
                self._streak = 0
                self._frames_since_cut = 0
                self._freeze_state = False
                return self.CUT_CONF
            return self.CUT_PEND
        else:
            if self._streak > 0:
                # Was in a streak, now it broke
                self._streak = 0
                self._freeze_state = False
                return self.CUT_ABORT
            self._freeze_state = False
            return self.CUT_NO

    @property
    def _should_update_state(self) -> bool:
        return not getattr(self, '_freeze_state', False)

    def process_frame(self, y_plane_cuda) -> bool:
        """Process a frame and return True if scene cut detected.

        Args:
            y_plane_cuda: cupy array of Y plane (luma) from NV12 CUDA frame

        Returns:
            True if scene cut detected, False otherwise
        """
        raise NotImplementedError("Subclasses must implement process_frame")





class HistogramDetector(SceneDetector):
    """Histogram correlation-based scene detector.
    
    Computes normalized luma histograms and detects scene changes when
    correlation between consecutive frames drops below threshold.
    """
    
    def process_frame(self, y_plane_cuda, uv_plane_cuda=None) -> str:
        hist = self._compute_histogram(y_plane_cuda)
        over = False
        if self._last_state is not None:
            corr = self._histogram_correlation(self._last_state, hist)
            over = corr < self.threshold
        result = self._check_cut(over)
        if self._should_update_state:
            self._last_state = hist
        return result
    
    def _compute_histogram(self, y_plane):
        """Compute normalized histogram on GPU."""
        hist = cp.histogram(y_plane, bins=self.hist_bins, range=(0, 256))[0]
        hist_norm = hist.astype(cp.float32) / hist.sum()
        return hist_norm
    
    def _histogram_correlation(self, hist1, hist2):
        """Compute correlation between two histograms."""
        mean1 = hist1.mean()
        mean2 = hist2.mean()
        
        numerator = ((hist1 - mean1) * (hist2 - mean2)).sum()
        denom1 = cp.sqrt(((hist1 - mean1) ** 2).sum())
        denom2 = cp.sqrt(((hist2 - mean2) ** 2).sum())
        
        if denom1 == 0 or denom2 == 0:
            return 1.0
        
        corr = numerator / (denom1 * denom2)
        return float(corr)


class ContentDetector(SceneDetector):
    """CUDA content-based scene detector, mirroring PySceneDetect's ContentDetector.

    Converts each NV12 CUDA frame to HSV entirely on GPU, then computes
    per-channel (H, S, V) histograms.  The content score is the weighted mean
    of the absolute histogram differences between consecutive frames.  A cut is
    declared when the score exceeds ``threshold``.

    Weights match PySceneDetect defaults: H=1.0, S=1.0, V=1.0 (equal).
    Override with ``weights`` = (h, s, v) and ``luma_only=True`` to use only V.

    Args:
        threshold:      score cut-point (default 27.0, same as PySceneDetect)
        min_scene_len:  minimum frames between cuts
        hist_bins:      histogram bins per channel (default 256)
        weights:        (h_weight, s_weight, v_weight) tuple, default (1,1,1)
        luma_only:      if True, use only the V (luma-like) channel
    """

    def __init__(self, threshold: float = 27.0, min_scene_len: int = 15,
                 hist_bins: int = 256,
                 weights: tuple = (1.0, 1.0, 1.0),
                 luma_only: bool = False, confirm_frames: int = 3):
        super().__init__(threshold=threshold, min_scene_len=min_scene_len,
                         hist_bins=hist_bins, confirm_frames=confirm_frames)
        self._weights = weights
        self._luma_only = luma_only
        # _last_state holds per-channel histograms: list of 3 cupy arrays
        self._last_state = None

    def process_frame(self, y_plane_cuda, uv_plane_cuda=None) -> bool:
        hists = self._compute_hsv_histograms(y_plane_cuda, uv_plane_cuda)

        over = False
        if self._last_state is not None:
            score = self._content_score(self._last_state, hists)
            over = score >= self.threshold
        result = self._check_cut(over)
        if self._should_update_state:
            self._last_state = hists
        return result

    def _compute_hsv_histograms(self, y_plane, uv_plane):
        """Convert NV12 → HSV on GPU and return per-channel histograms."""
        h, w = y_plane.shape

        # ── Y → float [0,1] ──────────────────────────────────────────────────
        Y = y_plane.astype(cp.float32) / 255.0

        if uv_plane is not None and not self._luma_only:
            # NV12 UV plane: height/2 rows, width bytes (interleaved U,V)
            uv_h, uv_stride = uv_plane.shape
            uv_w = min(w, uv_stride)  # valid columns

            # Upsample UV 2× to match Y resolution
            U = uv_plane[:uv_h, 0:uv_w:2].astype(cp.float32) / 255.0 - 0.5
            V = uv_plane[:uv_h, 1:uv_w:2].astype(cp.float32) / 255.0 - 0.5
            # repeat each UV row twice (nearest-neighbour vertical upsample)
            U = cp.repeat(U, 2, axis=0)[:h, :]
            V = cp.repeat(V, 2, axis=0)[:h, :]
            # horizontal: repeat each column twice
            U = cp.repeat(U, 2, axis=1)[:h, :w]
            V = cp.repeat(V, 2, axis=1)[:h, :w]

            # YUV → RGB (BT.601)
            R = cp.clip(Y + 1.402 * V,            0.0, 1.0)
            G = cp.clip(Y - 0.344136 * U - 0.714136 * V, 0.0, 1.0)
            B = cp.clip(Y + 1.772 * U,            0.0, 1.0)
        else:
            # No chroma — treat frame as greyscale (R=G=B=Y)
            R = G = B = Y

        # ── RGB → HSV ─────────────────────────────────────────────────────────
        Cmax = cp.maximum(cp.maximum(R, G), B)
        Cmin = cp.minimum(cp.minimum(R, G), B)
        delta = Cmax - Cmin

        # Value
        V_ch = Cmax

        # Saturation
        S_ch = cp.where(Cmax > 0, delta / Cmax, cp.zeros_like(Cmax))

        # Hue (0–360)
        eps = 1e-6
        H_ch = cp.zeros_like(Cmax)
        m_r = (Cmax == R) & (delta > eps)
        m_g = (Cmax == G) & (delta > eps)
        m_b = (Cmax == B) & (delta > eps)
        H_ch[m_r] = (60.0 * ((G[m_r] - B[m_r]) / delta[m_r])) % 360.0
        H_ch[m_g] = (60.0 * ((B[m_g] - R[m_g]) / delta[m_g]) + 120.0) % 360.0
        H_ch[m_b] = (60.0 * ((R[m_b] - G[m_b]) / delta[m_b]) + 240.0) % 360.0

        bins = self.hist_bins
        h_hist = cp.histogram(H_ch, bins=bins, range=(0.0, 360.0))[0].astype(cp.float32)
        s_hist = cp.histogram(S_ch, bins=bins, range=(0.0, 1.0))[0].astype(cp.float32)
        v_hist = cp.histogram(V_ch, bins=bins, range=(0.0, 1.0))[0].astype(cp.float32)

        # Normalize
        n = float(h * w)
        return [h_hist / n, s_hist / n, v_hist / n]

    def _content_score(self, prev_hists, curr_hists) -> float:
        """Weighted mean absolute histogram difference — the PySceneDetect score."""
        wh, ws, wv = self._weights
        total_w = wh + ws + wv
        if total_w == 0:
            return 0.0

        # Each channel: sum of |delta| * bins → scale to [0, 100]
        scores = []
        for prev, curr, w in zip(prev_hists, curr_hists, self._weights):
            if w == 0:
                continue
            # MAD * bins gives a value ∈ [0, 2]; multiply by 50 → [0, 100]
            mad = float(cp.sum(cp.abs(prev - curr)))
            scores.append(w * mad * 50.0)

        return sum(scores) / total_w


class AdaptiveDetector(ContentDetector):
    """CUDA adaptive scene detector mirroring PySceneDetect's AdaptiveDetector.

    Computes per-frame content scores (exactly like ContentDetector) then applies
    a rolling window: a cut fires when the centre-frame score divided by the mean
    of its neighbours exceeds ``adaptive_threshold`` AND the raw score exceeds
    ``min_content_val``.  This suppresses false positives from camera motion or
    gradual lighting changes that fool a plain threshold.

    Args:
        adaptive_threshold: ratio cut-point (default 3.0, same as PySceneDetect)
        min_scene_len:      minimum frames between cuts (default 15)
        window_width:       frames on each side of the candidate frame to average
                            (default 2, so the window covers 2*w+1 = 5 frames)
        min_content_val:    minimum raw content score for a cut to be considered
                            (default 15.0)
        hist_bins:          bins per HSV channel (default 256)
        weights:            (h, s, v) channel weights (default (1,1,1))
        luma_only:          use only the V channel (default False)
    """

    def __init__(self, adaptive_threshold: float = 3.0, min_scene_len: int = 15,
                 window_width: int = 2, min_content_val: float = 15.0,
                 hist_bins: int = 256, weights: tuple = (1.0, 1.0, 1.0),
                 luma_only: bool = False, confirm_frames: int = 3):
        super().__init__(threshold=255.0, min_scene_len=0,
                         hist_bins=hist_bins, weights=weights, luma_only=luma_only,
                         confirm_frames=confirm_frames)
        self.adaptive_threshold = adaptive_threshold
        self.min_content_val = min_content_val
        self.window_width = window_width
        self.min_scene_len = min_scene_len
        self._score_buffer: list = []
        self._frame_no_global: int = 0
        self._last_cut_frame: int = 0

    def process_frame(self, y_plane_cuda, uv_plane_cuda=None) -> bool:
        # Compute raw content score via parent (returns False — threshold=255)
        hists = self._compute_hsv_histograms(y_plane_cuda, uv_plane_cuda)
        if self._last_state is not None:
            score = self._content_score(self._last_state, hists)
        else:
            score = 0.0

        required = 1 + 2 * self.window_width
        self._score_buffer.append(score)
        self._frame_no_global += 1

        if len(self._score_buffer) < required:
            return False

        # Keep only the required window
        self._score_buffer = self._score_buffer[-required:]

        target_score = self._score_buffer[self.window_width]
        neighbour_scores = [s for i, s in enumerate(self._score_buffer)
                            if i != self.window_width]
        avg = sum(neighbour_scores) / (2.0 * self.window_width)

        if abs(avg) < 1e-5:
            adaptive_ratio = 255.0 if target_score >= self.min_content_val else 0.0
        else:
            adaptive_ratio = min(target_score / avg, 255.0)

        # The candidate frame is window_width frames behind the current one
        candidate_frame = self._frame_no_global - self.window_width - 1
        frames_since_cut = candidate_frame - self._last_cut_frame

        over = (adaptive_ratio >= self.adaptive_threshold
                and target_score >= self.min_content_val
                and frames_since_cut >= self.min_scene_len)
        result = self._check_cut(over)
        if self._should_update_state:
            self._last_state = hists
        if result == self.CUT_CONF:
            self._last_cut_frame = candidate_frame
        return result


class ThresholdDetector(SceneDetector):
    """Threshold-based scene detector using average pixel intensity.

    Mirrors PySceneDetect's ThresholdDetector: a scene cut is detected when
    the mean luma of a frame drops below ``threshold`` (fade to black) or
    rises back above it after being below (fade from black).  Typical values
    are in the range 10–30 (out of 255).

    Args:
        threshold:      mean-luma cut point (default 12, same as PySceneDetect)
        min_scene_len:  minimum frames between consecutive cuts
        fade_bias:      0.0 = detect both fade-in and fade-out (default),
                        negative = prefer fade-to-black,
                        positive = prefer fade-from-black
    """

    def __init__(self, threshold: float = 12.0, min_scene_len: int = 15,
                 hist_bins: int = 256, fade_bias: float = 0.0, confirm_frames: int = 3):
        super().__init__(threshold=threshold, min_scene_len=min_scene_len,
                         hist_bins=hist_bins, confirm_frames=confirm_frames)
        self._fade_bias = fade_bias
        self._last_mean: Optional[float] = None

    def process_frame(self, y_plane_cuda, uv_plane_cuda=None) -> bool:
        mean_luma = float(cp.mean(y_plane_cuda))
        threshold = self.threshold + self._fade_bias
        over = False
        if self._last_mean is not None:
            crossed_down = self._last_mean >= threshold > mean_luma
            crossed_up   = self._last_mean < threshold <= mean_luma
            over = crossed_down or crossed_up
        result = self._check_cut(over)
        if self._should_update_state:
            self._last_mean = mean_luma
        return result


class CudaSceneDetectNode(PythonNode):
    """CUDA-accelerated scene detection node for AVPlumber.
    
    Operates directly on NV12 CUDA frames. Detects scene changes and optionally
    writes metadata to frames for downstream overlay rendering.
    
    Args (node params):
        detector_type: Detection algorithm: "content", "adaptive", "histogram" or "threshold"
        threshold: Detection sensitivity (meaning depends on detector)
                  content:    weighted HSV histogram delta score, default 27.0
                  adaptive:   adaptive_threshold ratio, default 3.0
                  histogram:  luma histogram correlation, typically 0.90-0.98
                  threshold:  mean-luma cut point (0-255), typically 10-30
        min_scene_len: Minimum scene length in frames (default: 15)
        hist_bins: Histogram bins per channel (default: 256)
        fade_bias: Threshold detector only — bias toward fade-in/out (default: 0.0)
        luma_only: Content detector only — use only V channel (default: False)
        metadata_key: Optional key to write scene cut metadata to frames
        
    Outputs metadata in format:
        {
          "scene_count": int,
          "last_cut_time": float or None,
          "avg_process_time_ms": float
        }
    """
    
    DETECTOR_CLASSES = {
        "content":   ContentDetector,
        "adaptive":  AdaptiveDetector,
        "histogram": HistogramDetector,
        "threshold": ThresholdDetector,
    }
    
    def __init__(self, args):
        super().__init__(args)
        
        if not HAVE_CUPY:
            raise RuntimeError(
                "cupy is required for CUDA scene detection. "
                "Install with: pip install cupy-cuda12x"
            )
        
        # Extract parameters
        detector_type = args.get("detector_type", "content")
        _defaults = {"content": 27.0, "adaptive": 3.0, "threshold": 12.0, "histogram": 0.95}
        threshold = float(args.get("threshold", _defaults.get(detector_type, 27.0)))
        min_scene_len = int(args.get("min_scene_len", 15))
        hist_bins = int(args.get("hist_bins", 256))
        fade_bias = float(args.get("fade_bias", 0.0))
        luma_only = str(args.get("luma_only", "false")).lower() in ("1", "true", "yes")
        window_width = int(args.get("window_width", 2))
        min_content_val = float(args.get("min_content_val", 15.0))
        confirm_frames = int(args.get("confirm_frames", 3))
        self._metadata_key = args.get("metadata_key", "")
        self._needs_uv = (detector_type in ("content", "adaptive") and not luma_only)

        # Create detector
        if detector_type not in self.DETECTOR_CLASSES:
            raise ValueError(
                f"Unknown detector_type: {detector_type}. "
                f"Available: {list(self.DETECTOR_CLASSES.keys())}"
            )

        detector_cls = self.DETECTOR_CLASSES[detector_type]
        extra = {}
        if detector_type == "threshold":
            extra["fade_bias"] = fade_bias
        elif detector_type in ("content", "adaptive"):
            extra["luma_only"] = luma_only
        if detector_type == "adaptive":
            self._detector = AdaptiveDetector(
                adaptive_threshold=threshold,
                min_scene_len=min_scene_len,
                hist_bins=hist_bins,
                window_width=window_width,
                min_content_val=min_content_val,
                luma_only=luma_only,
                confirm_frames=confirm_frames,
            )
        else:
            self._detector = detector_cls(
                threshold=threshold,
                min_scene_len=min_scene_len,
                hist_bins=hist_bins,
                confirm_frames=confirm_frames,
                **extra
            )
        
        # State
        self._frame_no = 0
        self._cuts: List[Tuple[int, float]] = []
        self._process_time_total = 0.0
        self._last_activity = time.monotonic()
        self._pending_frames: list = []   # buffered frames during confirm streak
        
        thr_display = extra.get("adaptive_threshold", threshold)
        print(
            f"[CudaSceneDetect] initialized: type={detector_type}, "
            f"threshold={thr_display}, min_scene_len={min_scene_len}"
        )

        # Warm up cupy JIT kernels now, during __init__, so the first real frame
        # doesn't block for hundreds of ms inside process() and trigger a timeout.
        if detector_type in ("content", "adaptive"):
            self._warmup_cupy_hsv(hist_bins)
    
    def _warmup_cupy_hsv(self, hist_bins: int):
        """Trigger cupy JIT compilation for all HSV/histogram kernels during init."""
        import numpy as np
        try:
            y  = cp.asarray(np.zeros((64, 64), dtype=np.uint8))
            uv = cp.asarray(np.zeros((32, 64), dtype=np.uint8))
            self._detector.process_frame(y, uv)
            self._detector.reset() if hasattr(self._detector, 'reset') else None
            # Reset any state left by the warmup
            self._detector._last_state = None
            self._detector._frames_since_cut = 0
            if hasattr(self._detector, '_score_buffer'):
                self._detector._score_buffer = []
            if hasattr(self._detector, '_frame_no_global'):
                self._detector._frame_no_global = 0
                self._detector._last_cut_frame = 0
            print("[CudaSceneDetect] cupy HSV kernels warmed up")
        except Exception as e:
            print(f"[CudaSceneDetect] warmup warning: {e}")

    def _nv12_cuda_to_luma_cupy(self, frame):
        """Extract Y plane from NV12 CUDA frame as cupy array."""
        height = int(frame.height)
        width = int(frame.width)
        stride = int(frame.linesize[0]) if frame.linesize else 0
        ptr = int(frame.data_ptr[0]) if frame.data_ptr else 0

        if ptr <= 0 or stride <= 0 or width <= 0 or height <= 0:
            return None

        y_size = stride * height
        y_flat = cp.ndarray(
            shape=(y_size,),
            dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, y_size, owner=frame), 0)
        )
        return y_flat.reshape(height, stride)[:, :width]

    def _nv12_cuda_to_uv_cupy(self, frame):
        """Extract interleaved UV plane from NV12 CUDA frame as cupy array."""
        height = int(frame.height)
        width = int(frame.width)
        # UV plane: plane index 1, stride usually same as Y
        stride = int(frame.linesize[1]) if len(frame.linesize) > 1 else int(frame.linesize[0])
        ptr = int(frame.data_ptr[1]) if len(frame.data_ptr) > 1 else 0

        if ptr <= 0 or stride <= 0:
            return None

        uv_height = height // 2
        uv_size = stride * uv_height
        uv_flat = cp.ndarray(
            shape=(uv_size,),
            dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, uv_size, owner=frame), 0)
        )
        return uv_flat.reshape(uv_height, stride)[:, :width]
    
    def _frame_seconds(self, frame) -> float:
        """Extract timestamp in seconds from frame PTS."""
        NOPTS = -9223372036854775808
        pts = frame.pts
        ts = int(pts.timestamp)
        if ts == NOPTS:
            return float("nan")
        tb = pts.timebase
        if not tb or int(tb.den) == 0:
            return float(ts)
        return float(ts) * float(tb.num) / float(tb.den)
    
    def _fmt_overlay_ts(self, seconds):
        if seconds is None or seconds != seconds:
            return "NONE"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _write_metadata(self, frame):
        """Write scene detection metadata to frame if metadata_key is set."""
        if not self._metadata_key:
            return

        last_cut_time = self._cuts[-1][1] if self._cuts else None
        labels = [
            ("SCENES", f"{len(self._cuts) + 1:03d}", 40),
            ("LAST", self._fmt_overlay_ts(last_cut_time), 82),
        ]
        payload = {
            "coord_space": "frame",
            "detections": [
                {
                    "label": f"{prefix} {value}",
                    "conf": 1.0,
                    "xyxy": [24, y1, 300, y1 + 24],
                }
                for prefix, value, y1 in labels
            ],
        }
        frame.metadata[self._metadata_key] = json.dumps(payload)
    
    def _forward_if_configured(self, frame):
        """Forward frame to destination edges if configured."""
        dst = getattr(self, "_dst", None)
        if isinstance(dst, dict):
            for edge in dst.values():
                edge.enqueue(frame)
        elif dst is not None:
            dst.enqueue(frame)
    
    def process(self):
        """Process one frame (called repeatedly by AVPlumber event loop)."""
        frame = self._src.tryGet(1000)
        if frame is None:
            return

        self._last_activity = time.monotonic()
        t0 = time.perf_counter()

        y_plane = self._nv12_cuda_to_luma_cupy(frame)
        uv_plane = self._nv12_cuda_to_uv_cupy(frame) if self._needs_uv else None

        result = self._detector.CUT_NO
        if y_plane is not None:
            result = self._detector.process_frame(y_plane, uv_plane)

        self._process_time_total += time.perf_counter() - t0
        self._frame_no += 1

        if result == self._detector.CUT_PEND:
            # Hold this frame — streak is accumulating
            self._pending_frames.append(frame)

        elif result == self._detector.CUT_CONF:
            # Cut confirmed: the cut happened at the FIRST pending frame
            first = self._pending_frames[0] if self._pending_frames else frame
            cut_frame_no = self._frame_no - len(self._pending_frames) - 1
            seconds = self._frame_seconds(first)
            self._cuts.append((cut_frame_no, seconds))
            print(
                f"[CudaSceneDetect] scene change #{len(self._cuts)} "
                f"at frame {cut_frame_no} ({seconds:.3f}s)"
            )
            sys.stdout.flush()
            # Flush all pending frames (metadata already updated above via _write_metadata)
            for pf in self._pending_frames:
                self._write_metadata(pf)
                self._forward_if_configured(pf)
            self._pending_frames = []
            # Forward current frame too
            self._write_metadata(frame)
            self._forward_if_configured(frame)

        elif result == self._detector.CUT_ABORT:
            # Streak broke — flush buffered frames as normal (no cut)
            for pf in self._pending_frames:
                self._write_metadata(pf)
                self._forward_if_configured(pf)
            self._pending_frames = []
            self._write_metadata(frame)
            self._forward_if_configured(frame)

        else:
            # CUT_NO — forward immediately
            self._write_metadata(frame)
            self._forward_if_configured(frame)
    
    def idle_seconds(self) -> float:
        """Return seconds since last frame received."""
        return time.monotonic() - self._last_activity
    
    def get_stats(self) -> dict:
        """Return detection statistics."""
        avg_ms = (
            (self._process_time_total / self._frame_no) * 1000.0
            if self._frame_no > 0 else 0.0
        )
        return {
            "frames_analyzed": self._frame_no,
            "scene_changes": len(self._cuts),
            "scenes": len(self._cuts) + 1,
            "avg_frame_time_ms": round(avg_ms, 3),
            "cuts": self._cuts,
        }
    
    def finalize(self):
        """Print summary when processing ends."""
        # Flush any frames still buffered in a pending streak
        for pf in self._pending_frames:
            self._write_metadata(pf)
            self._forward_if_configured(pf)
        self._pending_frames = []

        stats = self.get_stats()
        
        print("\n=== CUDA Scene Detection Summary ===")
        print(f"frames analyzed     : {stats['frames_analyzed']}")
        print(f"scene changes       : {stats['scene_changes']}")
        print(f"scenes              : {stats['scenes']}")
        print(f"avg frame time      : {stats['avg_frame_time_ms']:.3f} ms")
        
        if not self._cuts:
            return
        
        print("\nscene boundaries:")
        start = 0.0
        for idx, (cut_frame, seconds) in enumerate(self._cuts, start=1):
            print(
                f"  scene {idx:<3} {self._fmt_ts(start)} -> "
                f"{self._fmt_ts(seconds)} (cut at frame {cut_frame})"
            )
            start = seconds
        print(f"  scene {len(self._cuts) + 1:<3} {self._fmt_ts(start)} -> EOF")
    
    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        """Format timestamp as HH:MM:SS.mmm"""
        if seconds != seconds:  # NaN
            return "??:??:??.???"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"
