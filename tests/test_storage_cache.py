import hashlib
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

from dotnxs.storage_cache import FileBackedTensorCache, StorageCacheError
from dotnxs.streamed_inkling import StreamedInklingRunner


class FileBackedTensorCacheTests(unittest.TestCase):
    def _entry(self, payload: bytes) -> dict:
        return {"nbytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    def test_materializes_verified_bytes_and_reuses_the_file(self):
        payload = b"NXS-backed weights"
        with tempfile.TemporaryDirectory() as directory:
            cache = FileBackedTensorCache(directory, max_bytes=1024, reserve_bytes=0)
            entry = self._entry(payload)
            path = cache.materialize(entry, [payload[:5], payload[5:]])
            self.assertEqual(path.read_bytes(), payload)
            reused = cache.materialize(entry, self.fail_if_consumed())
            self.assertEqual(reused, path)
            self.assertEqual(cache.stats(), {"limit_bytes": 1024, "used_bytes": len(payload), "hits": 1, "misses": 1, "evictions": 0})

    @staticmethod
    def fail_if_consumed():
        yield b"this generator must not be read on a cache hit"
        raise AssertionError("cache hit unexpectedly read the entry stream")

    def test_evicts_old_entries_to_respect_the_storage_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = FileBackedTensorCache(directory, max_bytes=8, reserve_bytes=0)
            first = b"12345678"
            second = b"abcdefgh"
            first_path = cache.materialize(self._entry(first), [first])
            cache.materialize(self._entry(second), [second])
            self.assertFalse(first_path.exists())
            self.assertEqual(cache.used_bytes, 8)
            self.assertEqual(cache.evictions, 1)

    def test_rejects_incomplete_or_corrupt_entry_without_leaving_a_cache_file(self):
        payload = b"valid bytes"
        with tempfile.TemporaryDirectory() as directory:
            cache = FileBackedTensorCache(directory, max_bytes=1024, reserve_bytes=0)
            with self.assertRaises(StorageCacheError):
                cache.materialize(self._entry(payload), [b"corrupt"])
            self.assertEqual(list(Path(directory).glob("*.tensor")), [])
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_rejects_an_entry_larger_than_the_storage_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = FileBackedTensorCache(directory, max_bytes=4, reserve_bytes=0)
            payload = b"12345"
            with self.assertRaises(StorageCacheError):
                cache.materialize(self._entry(payload), [payload])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "install PyTorch to test file-backed tensors")
    def test_torch_tensor_reads_from_the_cached_file(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            payload = struct.pack("<HH", 0x3F80, 0x4000)  # BF16 values 1 and 2
            cache = FileBackedTensorCache(directory, max_bytes=1024, reserve_bytes=0)
            path = cache.materialize(self._entry(payload), [payload])
            mapped = torch.from_file(str(path), shared=False, size=2, dtype=torch.bfloat16)
            self.assertEqual(mapped.tolist(), [1.0, 2.0])
            del mapped

    @unittest.skipUnless(importlib.util.find_spec("torch"), "install PyTorch to test mapped runner weights")
    def test_runner_installs_a_torch_parameter_from_a_mapped_nxs_entry(self):
        import torch

        payload = struct.pack("<HH", 0x3F80, 0x4000)
        entry = {
            "name": "test.weight",
            "dtype": "BF16",
            "shape": [2],
            "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

        class FakeArchive:
            cleared = False

            def get_entry(self, name):
                assert name == entry["name"]
                return entry

            def iter_entry_range(self, name, start, length):
                assert name == entry["name"] and start == 0 and length == len(payload)
                yield payload

            def clear_cache(self):
                self.cleared = True

        class FakeWeights:
            archive = FakeArchive()

        with tempfile.TemporaryDirectory() as directory:
            runner = object.__new__(StreamedInklingRunner)
            runner.torch = torch
            runner.weights = FakeWeights()
            runner.storage_cache = FileBackedTensorCache(directory, max_bytes=1024, reserve_bytes=0)
            runner._templates = {"test.weight": torch.empty((2,), dtype=torch.bfloat16)}
            tensor = runner._tensor_from_entry("test.weight", "test.weight")
            self.assertEqual(tensor.tolist(), [1.0, 2.0])
            self.assertFalse(runner.weights.archive.cleared)
            self.assertEqual(runner.storage_cache.used_bytes, len(payload))
            del tensor


if __name__ == "__main__":
    unittest.main()
