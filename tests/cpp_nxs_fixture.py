"""Create and serve a tiny NXS fixture for the native C++ reader smoke test."""

from __future__ import annotations

import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotnxs.archive import NxsWriter


class RangeHandler(BaseHTTPRequestHandler):
    payload = b""

    def _range(self, send_body: bool) -> None:
        header = self.headers.get("Range", "")
        if not header.startswith("bytes="):
            start, end, status = 0, len(self.payload) - 1, 200
        else:
            first, last = header.removeprefix("bytes=").split("-", 1)
            start, end, status = int(first), min(int(last), len(self.payload) - 1), 206
        body = self.payload[start : end + 1]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.payload)}")
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def do_HEAD(self):
        self._range(False)

    def do_GET(self):
        self._range(True)

    def log_message(self, format_string, *args):
        pass


def create(path: Path) -> None:
    with NxsWriter(path, chunk_bytes=4) as writer:
        writer.add_bytes("payload", b"abcdefgh", kind="asset")
        writer.add_bytes("tensor", b"0123456789abcdef", kind="tensor", dtype="U8", shape=[2, 8])


def serve(path: Path, port: int) -> None:
    RangeHandler.payload = path.read_bytes()
    ThreadingHTTPServer(("127.0.0.1", port), RangeHandler).serve_forever()


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "create":
        create(Path(sys.argv[2]))
    elif action == "serve":
        serve(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit("usage: cpp_nxs_fixture.py create PATH | serve PATH PORT")
