import tempfile
import threading
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dotnxs.archive import NxsArchive, NxsWriter
from dotnxs.http_range import HTTPRangeReader


class _RangeHandler(BaseHTTPRequestHandler):
    payload = bytes(range(256)) * 4
    requests = []

    def do_GET(self):
        value = self.headers.get("Range", "")
        if not value.startswith("bytes="):
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
            return
        start_text, end_text = value.removeprefix("bytes=").split("-", 1)
        start = int(start_text)
        end = min(int(end_text), len(self.payload) - 1)
        self.requests.append((start, end))
        body = self.payload[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.payload)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        pass


class HttpRangeReaderTests(unittest.TestCase):
    def setUp(self):
        _RangeHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/model.bin"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_seek_and_read_fetch_only_ranges(self):
        with HTTPRangeReader(self.url, len(_RangeHandler.payload), read_ahead_bytes=32) as reader:
            self.assertEqual(reader.read(5), _RangeHandler.payload[:5])
            reader.seek(20)
            self.assertEqual(reader.read(8), _RangeHandler.payload[20:28])
            reader.seek(-6, 2)
            self.assertEqual(reader.read(20), _RangeHandler.payload[-6:])
        self.assertLess(len(_RangeHandler.requests), 4)

    def test_rejects_unbounded_reads(self):
        with HTTPRangeReader(self.url, len(_RangeHandler.payload)) as reader:
            with self.assertRaisesRegex(ValueError, "unbounded reads"):
                reader.read()

    def test_nxs_writer_streams_a_remote_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "range.nxs"
            with NxsWriter(output, chunk_bytes=31) as writer:
                with HTTPRangeReader(
                    self.url,
                    len(_RangeHandler.payload),
                    read_ahead_bytes=64,
                ) as reader:
                    writer.add_stream(
                        "payload",
                        reader,
                        len(_RangeHandler.payload),
                        kind="asset",
                    )
            with NxsArchive(output) as archive:
                archive.verify()
                self.assertEqual(
                    b"".join(archive.iter_entry("payload")),
                    _RangeHandler.payload,
                )


if __name__ == "__main__":
    unittest.main()
