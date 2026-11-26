"""Tiny standard-library worker used only to smoke-test the C++ SSE proxy."""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path == "/health":
            body = b'{"status":"ready","input_protocol":"token_ids_v1","model_revision":"679300b3131ca739554d472b19f16dd714efaff0"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path != "/v1/chat-stream":
            self.send_error(404)
            return
        payload = parse_qs(parsed.query).get("payload", [""])[0]
        try:
            request = json.loads(payload)
        except json.JSONDecodeError:
            self.send_error(400, "invalid payload")
            return
        input_ids = request.get("input_ids")
        if not isinstance(input_ids, list) or not input_ids or any(type(token_id) is not int for token_id in input_ids):
            self.send_error(422, "missing input_ids")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for event in (
            b'event: status\ndata: {"status":"generating"}\n\n',
            b'event: token\ndata: {"id":42}\n\n',
            b'event: done\ndata: {}\n\n',
        ):
            self.wfile.write(f"{len(event):X}\r\n".encode("ascii") + event + b"\r\n")
            self.wfile.flush()
            time.sleep(0.02)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def log_message(self, format_string, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1])
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
