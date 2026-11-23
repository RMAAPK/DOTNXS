import importlib.util
import json
import struct
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotnxs.archive import NxsWriter
from dotnxs.remote_archive import RemoteNxsArchive


class _NxsRangeHandler(BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self):
        header = self.headers.get("Range", "")
        if not header.startswith("bytes="):
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
            return
        start_text, end_text = header.removeprefix("bytes=").split("-", 1)
        start = int(start_text)
        end = min(int(end_text), len(self.payload) - 1)
        body = self.payload[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.payload)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        pass


class RemoteNxsArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.archive_path = Path(self.temp_dir.name) / "model.nxs"
        self.server = None
        self.server_thread = None

    def tearDown(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server_thread.join(timeout=2)
        self.temp_dir.cleanup()

    def _serve_archive(self):
        _NxsRangeHandler.payload = self.archive_path.read_bytes()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _NxsRangeHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        return f"http://127.0.0.1:{self.server.server_port}/model.nxs", len(_NxsRangeHandler.payload)

    def _write_small_nxs(self, *, chunk_bytes=64):
        with NxsWriter(self.archive_path, chunk_bytes=chunk_bytes) as writer:
            writer.add_bytes("payload", b"abcdefgh", kind="asset")
            writer.add_bytes("tensor", b"0123456789abcdef", kind="tensor", dtype="U8", shape=[2, 8])

    def test_remote_reads_verify_and_evict_internal_chunks(self):
        self._write_small_nxs(chunk_bytes=4)
        url, file_size = self._serve_archive()
        with RemoteNxsArchive(url, file_size, cache_bytes=4) as archive:
            self.assertEqual(archive.read_entry("payload"), b"abcdefgh")
            stats = archive.cache_stats
            self.assertEqual(stats["used_bytes"], 4)
            self.assertGreaterEqual(stats["evictions"], 1)
            self.assertEqual(archive.read_tensor_expert("tensor", 1, 2), b"89abcdef")
            self.assertLessEqual(archive.cache_stats["used_bytes"], 4)

    def test_prefetch_parallelizes_and_caches_internal_chunks(self):
        self._write_small_nxs(chunk_bytes=4)
        url, file_size = self._serve_archive()
        with RemoteNxsArchive(url, file_size, cache_bytes=8) as archive:
            result = archive.prefetch_entry_ranges([("payload", 0, 8)], max_workers=2)
            self.assertEqual(result, {"ranges": 2, "bytes": 8})
            before_read = archive.cache_stats
            self.assertEqual(before_read["used_bytes"], 8)
            self.assertEqual(archive.read_entry("payload"), b"abcdefgh")
            after_read = archive.cache_stats
            self.assertEqual(after_read["network_ranges"], before_read["network_ranges"])
            self.assertEqual(after_read["used_bytes"], 8)
            self.assertGreaterEqual(after_read["hits"], 2)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "install dotnxs[runtime] for CPU inference test")
    def test_real_sparse_moe_path_over_a_local_nxs_archive(self):
        import numpy as np

        config = {
            "model_type": "inkling2_loop",
            "text_config": {
                "num_hidden_layers": 1,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "hidden_size": 16,
                "intermediate_size": 16,
                "vocab_size": 4,
                "unpadded_vocab_size": 3,
                "rms_norm_eps": 1e-6,
                "final_logit_softcapping": 30.0,
                "route_scale": 8.0,
                "mlp_gate_softcap": 4.0,
                "mlp_up_softcap": 25.0,
            },
        }
        def as_bf16_bytes(values):
            floats = np.asarray(values, dtype=np.float32)
            return (floats.view(np.uint32) >> 16).astype("<u2").tobytes()

        embedding = np.vstack([np.zeros(16), np.ones(16), np.arange(16), np.full(16, 2)])
        unembedding = np.vstack([np.ones(16), np.full(16, 2), np.arange(16), np.full(16, 3)])
        with NxsWriter(self.archive_path, chunk_bytes=64) as writer:
            writer.add_bytes("asset/config.json", json.dumps(config).encode(), kind="asset")
            writer.add_bytes(
                "model.llm.embed.weight", as_bf16_bytes(embedding),
                kind="tensor", dtype="BF16", shape=[4, 16]
            )
            writer.add_bytes(
                "model.llm.embed_norm.weight", as_bf16_bytes(np.ones(16)),
                kind="tensor", dtype="BF16", shape=[16]
            )
            writer.add_bytes(
                "model.llm.norm.weight", as_bf16_bytes(np.ones(16)),
                kind="tensor", dtype="BF16", shape=[16]
            )
            writer.add_bytes(
                "model.llm.unembed.weight", as_bf16_bytes(unembedding),
                kind="tensor", dtype="BF16", shape=[4, 16]
            )
            base = "model.llm.layers.0.mlp."
            experts = base + "experts."
            writer.add_bytes(base + "gate.weight", bytes(2 * 16 * 2), kind="tensor", dtype="BF16", shape=[2, 16])
            writer.add_bytes(
                base + "gate.bias", struct.pack("<2f", 1.0, 0.0), kind="tensor", dtype="F32", shape=[2]
            )
            writer.add_bytes(
                experts + "w13_weight", bytes([0x11]) * (2 * 32 * 8),
                kind="tensor", dtype="U8", shape=[2, 32, 8]
            )
            writer.add_bytes(
                experts + "w13_weight_scale", bytes([0x38]) * (2 * 32),
                kind="tensor", dtype="F8_E4M3", shape=[2, 32, 1]
            )
            writer.add_bytes(
                experts + "w13_weight_amax", struct.pack("<2f", 1.0, 1.0),
                kind="tensor", dtype="F32", shape=[2]
            )
            writer.add_bytes(
                experts + "w13_weight_outer_scale_denominator", struct.pack("<f", 1.0),
                kind="tensor", dtype="F32", shape=[1]
            )
            writer.add_bytes(
                experts + "w2_weight", bytes([0x11]) * (2 * 16 * 8),
                kind="tensor", dtype="U8", shape=[2, 16, 8]
            )
            writer.add_bytes(
                experts + "w2_weight_scale", bytes([0x38]) * (2 * 16),
                kind="tensor", dtype="F8_E4M3", shape=[2, 16, 1]
            )
            writer.add_bytes(
                experts + "w2_weight_amax", struct.pack("<2f", 1.0, 1.0),
                kind="tensor", dtype="F32", shape=[2]
            )
            writer.add_bytes(
                experts + "w2_weight_outer_scale_denominator", struct.pack("<f", 1.0),
                kind="tensor", dtype="F32", shape=[1]
            )

        url, file_size = self._serve_archive()
        with RemoteNxsArchive(url, file_size, cache_bytes=1024 * 1024) as archive:
            from dotnxs.inkling_runtime import Inkling2MiniWeights

            weights = Inkling2MiniWeights(archive)
            output = weights.run_moe_layer(0, np.ones(16, dtype=np.float32))
            self.assertEqual(output.shape, (16,))
            self.assertTrue(np.isfinite(output).all())
            self.assertGreater(float(np.abs(output).sum()), 0.0)

            embedding_row = weights.embed_token(1)
            np.testing.assert_array_equal(embedding_row, np.ones(16, dtype=np.float32))
            logits = weights.project_logits(np.ones(16, dtype=np.float32), block_rows=2)
            self.assertEqual(logits.shape, (3,))
            final_hidden = np.ones(16, dtype=np.float32)
            from dotnxs.inkling_kernels import round_bf16

            raw_logits = round_bf16(unembedding[:3] @ final_hidden)
            expected_logits = 30.0 * np.tanh(raw_logits / 30.0)
            np.testing.assert_allclose(logits, expected_logits, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
