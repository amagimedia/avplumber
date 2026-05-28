from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import urllib.error
import urllib.request


JANUS_REST = os.environ.get(
    "JANUS_REST",
    f"http://127.0.0.1:{os.environ.get('JANUS_HTTP_PORT', '8088')}/janus",
)
PREVIEW_PORT = int(os.environ.get("JANUS_PREVIEW_PORT", "8080"))


class PreviewHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.startswith("/janus"):
            self._proxy_janus()
            return
        super().do_GET()

    def do_POST(self):
        if self.path.startswith("/janus"):
            self._proxy_janus()
            return
        self.send_error(404, "File not found")

    def _proxy_janus(self):
        suffix = self.path[len("/janus"):]
        upstream_url = f"{JANUS_REST}{suffix}"
        body = None
        if self.command in ("POST", "PUT", "PATCH"):
            length = int(self.headers.get("content-length", "0") or "0")
            body = self.rfile.read(length) if length else b""
        headers = {
            "content-type": self.headers.get("content-type", "application/json"),
        }
        request = urllib.request.Request(
            upstream_url,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=65) as response:
                payload = response.read()
                self.send_response(response.status)
                self.send_header(
                    "Content-Type",
                    response.headers.get("content-type", "application/json"),
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self.send_header("Content-Type", error.headers.get("content-type", "text/plain"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as error:
            payload = str(error).encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PREVIEW_PORT), PreviewHandler).serve_forever()
