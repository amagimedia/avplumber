"""Persistent Molmo2 Transformers sidecar and AVPlumber client runner."""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

import numpy as np

try:
    from pyplumber.molmo_transformers_runner import (
        TransformersMolmo2VideoRunner,
        _get_float,
        _get_int,
        _metadata_for_transformers_window,
        window_to_video_array,
    )
except ModuleNotFoundError:
    from molmo_transformers_runner import (  # type: ignore
        TransformersMolmo2VideoRunner,
        _get_float,
        _get_int,
        _metadata_for_transformers_window,
        window_to_video_array,
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        value = value.replace(";", ",")
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    raise TypeError(f"expected string or list, got {type(value).__name__}")


def _encode_request(
    *,
    prompt: str,
    video: Any,
    metadata: dict[str, Any],
    max_new_tokens: int | None,
    compressed: bool = False,
) -> bytes:
    output = io.BytesIO()
    save = np.savez_compressed if compressed else np.savez
    arrays: dict[str, Any] = {
        "prompt": np.array(prompt),
        "metadata": np.array(json.dumps(metadata, separators=(",", ":"))),
        "video": video,
    }
    if max_new_tokens is not None:
        arrays["max_new_tokens"] = np.array(int(max_new_tokens), dtype=np.int32)
    save(output, **arrays)
    return output.getvalue()


def _decode_request(body: bytes) -> tuple[str, Any, dict[str, Any], int | None]:
    with np.load(io.BytesIO(body), allow_pickle=False) as payload:
        prompt = str(payload["prompt"].item())
        metadata = json.loads(str(payload["metadata"].item()))
        video = payload["video"]
        max_new_tokens = None
        if "max_new_tokens" in payload:
            max_new_tokens = int(payload["max_new_tokens"].item())
    return prompt, video, metadata, max_new_tokens


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class SidecarRequestHandler(BaseHTTPRequestHandler):
    server: "MolmoSidecarServer"

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.server.quiet:
            return
        super().log_message(fmt, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self._send_json(404, {"error": "not_found"})
            return
        self._send_json(
            200,
            {
                "ok": True,
                "model_id": self.server.runner.model_id,
                "loaded": self.server.runner._model is not None,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._send_json(404, {"error": "not_found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("empty request body")
            prompt, video, metadata, max_new_tokens = _decode_request(self.rfile.read(content_length))

            start = time.perf_counter()
            with self.server.model_lock:
                old_max_new_tokens = self.server.runner.max_new_tokens
                if max_new_tokens is not None:
                    self.server.runner.max_new_tokens = max_new_tokens
                try:
                    generated_text = self.server.runner.generate_from_video(
                        prompt=prompt,
                        video=video,
                        metadata=metadata,
                    )
                finally:
                    self.server.runner.max_new_tokens = old_max_new_tokens
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._send_json(200, {"generated_text": generated_text, "latency_ms": latency_ms})
        except Exception as exc:
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})


class MolmoSidecarServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        runner: TransformersMolmo2VideoRunner,
        *,
        quiet: bool,
    ) -> None:
        super().__init__(server_address, SidecarRequestHandler)
        self.runner = runner
        self.model_lock = Lock()
        self.quiet = quiet


class SidecarMolmo2VideoRunner:
    """Generate Molmo2 text through a persistent local sidecar process."""

    def __init__(self, config: dict[str, Any]):
        self.frame_size = _get_int(config, "frame_size", 378)
        self.patch_size = _get_int(config, "patch_size", 14)
        self.sample_fps = _get_float(config, "sample_fps", 0.5)
        self.max_new_tokens = _get_int(config, "max_new_tokens", 512)
        configured_urls = _string_list(config.get("sidecar_urls"))
        env_urls = _string_list(os.environ.get("AVP_MOLMO_SIDECAR_URLS"))
        single_url = str(
            config.get("sidecar_url")
            or os.environ.get("AVP_MOLMO_SIDECAR_URL")
            or "http://127.0.0.1:8765/generate"
        ).strip()
        self.urls = configured_urls or env_urls or [single_url]
        if not self.urls:
            raise ValueError("at least one Molmo sidecar URL is required")
        self.url = self.urls[0]
        self._url_index = 0
        self._url_lock = Lock()
        self.timeout_seconds = _get_float(
            config,
            "sidecar_timeout_seconds",
            float(os.environ.get("AVP_MOLMO_SIDECAR_TIMEOUT_SECONDS", "300")),
        )
        self.compressed = _env_bool("AVP_MOLMO_SIDECAR_COMPRESS", False)

    def _next_url(self) -> str:
        with self._url_lock:
            url = self.urls[self._url_index % len(self.urls)]
            self._url_index += 1
            return url

    def _post(self, body: bytes) -> dict[str, Any]:
        url = self._next_url()
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Molmo sidecar HTTP {exc.code}: {detail}") from exc

    def generate(self, job: Any) -> str:
        video = window_to_video_array(job, frame_size=self.frame_size, patch_size=self.patch_size)
        metadata = _metadata_for_transformers_window(
            sample_count=int(job.sample_count),
            frame_size=self.frame_size,
            sample_fps=self.sample_fps,
        )
        body = _encode_request(
            prompt=str(job.prompt),
            video=video,
            metadata=metadata,
            max_new_tokens=self.max_new_tokens,
            compressed=self.compressed,
        )
        payload = self._post(body)
        if "error" in payload:
            raise RuntimeError(str(payload["error"]))
        return str(payload.get("generated_text", ""))


def create_runner(config: dict[str, Any]) -> SidecarMolmo2VideoRunner:
    return SidecarMolmo2VideoRunner(config)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent Molmo2 Transformers sidecar")
    parser.add_argument("--host", default=os.environ.get("AVP_MOLMO_SIDECAR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AVP_MOLMO_SIDECAR_PORT", "8765")))
    parser.add_argument("--model-id", default=os.environ.get("AVP_MOLMO_MODEL_ID", "allenai/Molmo2-8B"))
    parser.add_argument("--model-dtype", default=os.environ.get("AVP_MOLMO_MODEL_DTYPE", "auto"))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("AVP_MOLMO_MAX_NEW_TOKENS", "512")))
    parser.add_argument("--prompt-style", default=os.environ.get("AVP_MOLMO_PROMPT_STYLE", ""))
    parser.add_argument("--preload", action="store_true", default=_env_bool("AVP_MOLMO_SIDECAR_PRELOAD", True))
    parser.add_argument("--quiet", action="store_true", default=_env_bool("AVP_MOLMO_SIDECAR_QUIET", False))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    runner = TransformersMolmo2VideoRunner(
        {
            "model_id": args.model_id,
            "model_dtype": args.model_dtype,
            "max_new_tokens": args.max_new_tokens,
            "prompt_style": args.prompt_style,
        }
    )
    if args.preload:
        runner._ensure_model()
    server = MolmoSidecarServer((args.host, args.port), runner, quiet=args.quiet)
    print(
        "molmo_transformers_sidecar_listening",
        _json_bytes({"host": args.host, "port": args.port, "model_id": args.model_id}).decode("utf-8"),
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
