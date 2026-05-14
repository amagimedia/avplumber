import json
import math
import os
from pathlib import Path

import _avplumber
from .node import AudioToMetadataPythonNode, PythonNode


def _ms_to_samples(ms: float, sample_rate: int) -> int:
    return max(0, int(round(float(ms) * sample_rate / 1000.0)))


def _audio_samples_to_float32_mono(samples):
    import numpy as np

    fmt = samples.sampleFormatName
    channels = int(samples.channelsCount)
    count = int(samples.samplesCount)
    planes = samples.planes
    if count <= 0 or channels <= 0:
        return np.empty(0, dtype=np.float32)

    if fmt == "flt":
        arr = np.frombuffer(planes[0], dtype=np.float32, count=count * channels)
        return np.ascontiguousarray(arr.reshape(count, channels).mean(axis=1), dtype=np.float32)
    if fmt == "fltp":
        if channels == 1:
            return np.ascontiguousarray(np.frombuffer(planes[0], dtype=np.float32, count=count), dtype=np.float32)
        stacked = [np.frombuffer(planes[i], dtype=np.float32, count=count) for i in range(channels)]
        return np.ascontiguousarray(np.mean(stacked, axis=0), dtype=np.float32)
    if fmt == "s16":
        arr = np.frombuffer(planes[0], dtype=np.int16, count=count * channels)
        return np.ascontiguousarray(arr.reshape(count, channels).mean(axis=1) / 32768.0, dtype=np.float32)
    if fmt == "s16p":
        stacked = [np.frombuffer(planes[i], dtype=np.int16, count=count).astype(np.float32) for i in range(channels)]
        return np.ascontiguousarray(np.mean(stacked, axis=0) / 32768.0, dtype=np.float32)
    if fmt == "s32":
        arr = np.frombuffer(planes[0], dtype=np.int32, count=count * channels)
        return np.ascontiguousarray(arr.reshape(count, channels).mean(axis=1) / 2147483648.0, dtype=np.float32)
    if fmt == "s32p":
        stacked = [np.frombuffer(planes[i], dtype=np.int32, count=count).astype(np.float32) for i in range(channels)]
        return np.ascontiguousarray(np.mean(stacked, axis=0) / 2147483648.0, dtype=np.float32)
    raise RuntimeError(f"unsupported VAD sample format: {fmt}")


def _float32_mono_rms_db(audio) -> float:
    if audio.size == 0:
        return -96.0
    import numpy as np

    audio = audio.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(audio * audio)))
    if rms <= 0.0:
        return -96.0
    return max(-96.0, 20.0 * math.log10(rms + 1e-12))


class SileroVADNode(AudioToMetadataPythonNode):
    def __init__(self, args: dict):
        super().__init__(args)
        p = self.parameters
        self.sample_rate = int(p.get("sample_rate", 16000))
        self.window_size_samples = int(p.get("window_size_samples", 512))
        self.threshold = float(p.get("threshold", 0.5))
        self.neg_threshold = float(p.get("neg_threshold", max(0.01, self.threshold - 0.15)))
        self.min_emit_samples = _ms_to_samples(p.get("min_emit_ms", 1200), self.sample_rate)
        self.min_silence_samples = _ms_to_samples(p.get("min_silence_ms", 450), self.sample_rate)
        self.speech_pad_samples = _ms_to_samples(p.get("speech_pad_ms", 120), self.sample_rate)
        self.max_discontinuity_ms = float(p.get("max_discontinuity_ms", 250))
        self.source_name = str(p.get("source", p.get("name", "audio")))
        self.device = str(p.get("device", "cpu"))
        self.model_path = p.get("model_path") or os.environ.get("AVP_SILERO_MODEL")
        self.repo_or_dir = p.get("repo_or_dir") or os.environ.get("AVP_SILERO_REPO", "snakers4/silero-vad")
        self.emit_state_events = bool(p.get("emit_state_events", False))
        self.emit_state_updates = bool(p.get("emit_state_updates", False))

        self._torch = None
        self._np = None
        self._model = None
        self._pending = None
        self._pending_start_pts = None
        self._expected_next_pts = None
        self._timebase = None
        self._stream_floor_pts = None
        self._reset_segment()

    def _ensure_model(self):
        if self._model is not None:
            return
        import numpy as np
        import torch

        self._np = np
        self._torch = torch
        if self.model_path:
            self._model = torch.jit.load(self.model_path, map_location=self.device)
        else:
            source = "local" if Path(self.repo_or_dir).exists() else "github"
            self._model, _ = torch.hub.load(
                repo_or_dir=self.repo_or_dir,
                model="silero_vad",
                source=source,
                trust_repo=True,
                onnx=False,
            )
            self._model.to(self.device)
        self._model.eval()
        self._reset_model_state()

    def _reset_model_state(self):
        if self._model is None or not hasattr(self._model, "reset_states"):
            return
        try:
            self._model.reset_states()
        except TypeError:
            self._model.reset_states(batch_size=1)

    def _reset_segment(self):
        self._in_speech = False
        self._segment_start_pts = None
        self._last_speech_pts = None
        self._silence_start_pts = None
        self._last_prob = 0.0
        self._last_level_db = -96.0

    def _reset_stream_state(self):
        self._pending = None
        self._pending_start_pts = None
        self._expected_next_pts = None
        self._stream_floor_pts = None
        self._reset_segment()
        self._reset_model_state()

    def doStart(self):
        self._ensure_model()
        super().doStart()

    def doStop(self):
        super().doStop()
        self.flush_open_segment()

    def _sample_count_to_pts(self, sample_count: int) -> int:
        tb = self._timebase
        if tb is None or tb.num == 0:
            return int(sample_count)
        return int(round(sample_count * tb.den / (self.sample_rate * tb.num)))

    def _pts_to_ms(self, pts_delta: int) -> float:
        tb = self._timebase
        if tb is None:
            return 1000.0 * pts_delta / self.sample_rate
        return 1000.0 * pts_delta * tb.num / tb.den

    def _pts_to_sec(self, pts: int) -> float:
        tb = self._timebase
        if tb is None:
            return pts / self.sample_rate
        return pts * tb.num / tb.den

    def _event_frame(self, pts: int, metadata: dict):
        return _avplumber.MetadataFrame(int(pts), self._timebase, metadata)

    def _enqueue_event(self, frame):
        if isinstance(self._dst, dict):
            for dst in self._dst.values():
                dst.enqueue(frame)
        else:
            self._dst.enqueue(frame)

    def _emit_state_event(self, event: str, pts: int, metadata: dict | None = None):
        if not self.emit_state_events:
            return
        tb = self._timebase
        payload = {
            "event": event,
            "source": self.source_name,
            "pts": int(pts),
            "sec": round(self._pts_to_sec(pts), 6),
            "timebase_num": int(tb.num) if tb else 1,
            "timebase_den": int(tb.den) if tb else self.sample_rate,
            "sample_rate": self.sample_rate,
            "threshold": self.threshold,
            "last_probability": round(float(self._last_prob), 6),
            "level_db": round(float(self._last_level_db), 3),
        }
        if metadata:
            payload.update(metadata)
        self._enqueue_event(self._event_frame(pts, payload))

    def _emit_segment(self, start_pts: int, end_pts: int):
        if end_pts <= start_pts:
            return
        duration_ms = self._pts_to_ms(end_pts - start_pts)
        if duration_ms < (1000.0 * self.min_emit_samples / self.sample_rate):
            return

        tb = self._timebase
        metadata = {
            "event": "speech",
            "source": self.source_name,
            "speech_start_pts": int(start_pts),
            "speech_end_pts": int(end_pts),
            "speech_start_sec": round(self._pts_to_sec(start_pts), 6),
            "speech_end_sec": round(self._pts_to_sec(end_pts), 6),
            "duration_ms": int(round(duration_ms)),
            "timebase_num": int(tb.num) if tb else 1,
            "timebase_den": int(tb.den) if tb else self.sample_rate,
            "sample_rate": self.sample_rate,
            "threshold": self.threshold,
            "last_probability": round(float(self._last_prob), 6),
        }
        self._enqueue_event(self._event_frame(end_pts, metadata))

    def _vad_probability(self, chunk):
        self._ensure_model()
        if not chunk.flags.writeable:
            chunk = chunk.copy()
        tensor = self._torch.from_numpy(chunk).to(self.device)
        with self._torch.inference_mode():
            result = self._model(tensor, self.sample_rate)
        return float(result.item())

    def _handle_window(self, chunk, chunk_start_pts: int, chunk_end_pts: int):
        prob = self._vad_probability(chunk)
        self._last_prob = prob
        self._last_level_db = _float32_mono_rms_db(chunk)
        if prob >= self.threshold:
            if not self._in_speech:
                pad_pts = self._sample_count_to_pts(self.speech_pad_samples)
                start_pts = chunk_start_pts - pad_pts
                if self._stream_floor_pts is not None:
                    start_pts = max(self._stream_floor_pts, start_pts)
                self._segment_start_pts = start_pts
                self._in_speech = True
                self._emit_state_event("speech_start", chunk_end_pts, {
                    "speech_start_pts": int(start_pts),
                    "speech_start_sec": round(self._pts_to_sec(start_pts), 6),
                })
            elif self.emit_state_updates:
                self._emit_state_event("speech_update", chunk_end_pts)
            self._last_speech_pts = chunk_end_pts
            self._silence_start_pts = None
            return

        if not self._in_speech or prob >= self.neg_threshold:
            return

        if self._silence_start_pts is None:
            self._silence_start_pts = chunk_start_pts
        silence_pts = chunk_end_pts - self._silence_start_pts
        if self._pts_to_ms(silence_pts) < (1000.0 * self.min_silence_samples / self.sample_rate):
            return

        pad_pts = self._sample_count_to_pts(self.speech_pad_samples)
        end_pts = min(chunk_end_pts, (self._last_speech_pts or self._silence_start_pts) + pad_pts)
        self._emit_segment(self._segment_start_pts, end_pts)
        self._emit_state_event("speech_stop", chunk_end_pts, {
            "speech_start_pts": int(self._segment_start_pts),
            "speech_end_pts": int(end_pts),
            "speech_start_sec": round(self._pts_to_sec(self._segment_start_pts), 6),
            "speech_end_sec": round(self._pts_to_sec(end_pts), 6),
            "duration_ms": int(round(self._pts_to_ms(end_pts - self._segment_start_pts))),
        })
        self._reset_segment()

    def _append_audio(self, audio, start_pts: int):
        if audio.size == 0:
            return
        if self._pending is None or self._pending.size == 0:
            self._pending = audio
            self._pending_start_pts = start_pts
            return
        self._pending = self._np.concatenate((self._pending, audio))

    def _process_pending(self):
        while self._pending is not None and self._pending.size >= self.window_size_samples:
            chunk = self._pending[:self.window_size_samples]
            chunk_start_pts = self._pending_start_pts
            chunk_end_pts = chunk_start_pts + self._sample_count_to_pts(chunk.size)
            self._handle_window(chunk, chunk_start_pts, chunk_end_pts)
            self._pending = self._pending[self.window_size_samples:]
            self._pending_start_pts = chunk_end_pts

    def flush_open_segment(self):
        if self._in_speech and self._segment_start_pts is not None and self._last_speech_pts is not None:
            self._emit_segment(self._segment_start_pts, self._last_speech_pts)
            self._emit_state_event("speech_stop", self._last_speech_pts, {
                "speech_start_pts": int(self._segment_start_pts),
                "speech_end_pts": int(self._last_speech_pts),
                "speech_start_sec": round(self._pts_to_sec(self._segment_start_pts), 6),
                "speech_end_sec": round(self._pts_to_sec(self._last_speech_pts), 6),
                "duration_ms": int(round(self._pts_to_ms(self._last_speech_pts - self._segment_start_pts))),
                "reason": "flush",
            })
        self._reset_segment()

    def process(self):
        samples = self._src.get()
        if not samples or int(samples.samplesCount) <= 0:
            self.flush_open_segment()
            return
        if int(samples.sampleRate) != self.sample_rate:
            raise RuntimeError(f"VAD expected {self.sample_rate} Hz audio, got {samples.sampleRate}")

        self._ensure_model()
        audio = _audio_samples_to_float32_mono(samples)
        pts = samples.pts
        if self._timebase is None:
            self._timebase = pts.timebase
        elif self._timebase.num != pts.timebase.num or self._timebase.den != pts.timebase.den:
            self.flush_open_segment()
            self._reset_stream_state()
            self._timebase = pts.timebase

        start_pts = int(pts.timestamp)
        if self._stream_floor_pts is None:
            self._stream_floor_pts = start_pts
        if self._expected_next_pts is not None:
            drift_ms = abs(self._pts_to_ms(start_pts - self._expected_next_pts))
            if drift_ms > self.max_discontinuity_ms:
                self.flush_open_segment()
                self._reset_stream_state()
                self._timebase = pts.timebase
                self._stream_floor_pts = start_pts

        self._append_audio(audio, start_pts)
        self._process_pending()
        self._expected_next_pts = start_pts + self._sample_count_to_pts(audio.size)


class MetadataJsonlSink(PythonNode):
    def __init__(self, args: dict):
        params = {"data_type": "MetadataFrame"} | args
        super().__init__(params)
        self.path = params.get("path")
        self._fh = None

    def _ensure_open(self):
        if self.path and self._fh is None:
            path = Path(self.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")

    def doStart(self):
        self._ensure_open()
        super().doStart()

    def doStop(self):
        super().doStop()
        if self._fh:
            self._fh.close()
            self._fh = None

    def process(self):
        frame = self._src.get()
        if not frame:
            return
        metadata = dict(frame.metadata.as_dict)
        if not metadata:
            return
        tb = frame.pts.timebase
        record = {
            "pts": int(frame.pts.timestamp),
            "timebase_num": int(tb.num),
            "timebase_den": int(tb.den),
            "metadata": metadata,
        }
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        self._ensure_open()
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()
